import org.cloudsimplus.allocationpolicies.VmAllocationPolicyFirstFit;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.listeners.DatacenterVmMigrationEventInfo;
import org.cloudsimplus.listeners.EventInfo;
import org.cloudsimplus.power.PowerMeasurement;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelAbstract;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * PatchTST-risk-aware Bitbrains replay for CloudSim Plus 8.5.7.
 * Host capacities are explicit simulation assumptions, not dataset facts.
 */
public class Main {
    private static final Path INPUT = Path.of("data", "bitbrains_cloudsim_predictions.csv");
    private static final Path OUT = Path.of(System.getProperty("sim.output.dir", "results"));
    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final int HOST_COUNT = Integer.getInteger("sim.host.count", 200);
    private static final int HOST_PES = Integer.getInteger("sim.host.pes", 32), VM_PES = 4;
    private static final int MIGRATION_PE_HEADROOM = 1;
    private static final double HOST_MIPS = 1000, VM_MIPS = 1000;
    private static final long HOST_RAM_MB = Long.getLong("sim.host.ram.mb", 262144L);
    private static final long HOST_BW = 1_000_000, HOST_STORAGE_MB = 10_000_000;
    private static final long VM_BW = 10_000, VM_STORAGE_MB = 20_000;
    private static final double HOST_CPU_LIMIT = Double.parseDouble(System.getProperty("sim.host.cpu.limit", "0.80"));
    private static final double HOST_RAM_LIMIT = Double.parseDouble(System.getProperty("sim.host.ram.limit", "0.80"));
    private static final double DECISION_INTERVAL = 900, COOLDOWN_SECONDS = 1800, STALE_SECONDS = 1800;
    private static final long CLOUDLET_LENGTH = 9_000_000_000_000_000L;

    private static final class Sample {
        final LocalDateTime timestamp;
        final double currentCpuPct, currentMemoryKB, forecastCpuMaxPct, forecastMemoryMaxKB, risk, threshold;
        final boolean alert;
        Sample(LocalDateTime timestamp, double currentCpuPct, double currentMemoryKB,
               double forecastCpuMaxPct, double forecastMemoryMaxKB, double risk, double threshold) {
            this.timestamp = timestamp;
            this.currentCpuPct = Math.max(0, currentCpuPct);
            this.currentMemoryKB = Math.max(0, currentMemoryKB);
            this.forecastCpuMaxPct = Math.max(0, forecastCpuMaxPct);
            this.forecastMemoryMaxKB = Math.max(0, forecastMemoryMaxKB);
            this.risk = clamp(risk, 0, 1);
            this.threshold = clamp(threshold, 0, 1);
            this.alert = this.risk >= this.threshold;
        }
        double cpuDemandPct() { return Math.max(currentCpuPct, forecastCpuMaxPct); }
        double ramDemandMB() { return Math.max(currentMemoryKB, forecastMemoryMaxKB) / 1024.0; }
    }

    private static final class Trace {
        final String sourceId;
        LocalDateTime firstTimestamp, lastTimestamp;
        Sample latestSample;
        long sampleCount, alertCount;
        double peakRamMB;
        Vm vm;
        long provisionedRamMB;
        Trace(String sourceId) { this.sourceId = sourceId; }
        void observe(Sample sample) {
            if (firstTimestamp == null) firstTimestamp = sample.timestamp;
            lastTimestamp = sample.timestamp;
            sampleCount++;
            if (sample.alert) alertCount++;
            peakRamMB = Math.max(peakRamMB, sample.ramDemandMB());
        }
        long peakRamMB() {
            return Math.max(2048, (long)(Math.ceil(Math.min(HOST_RAM_MB, peakRamMB * 1.20) / 1024.0) * 1024));
        }
    }

    private static final class PredictionReplay implements AutoCloseable {
        private final BufferedReader reader;
        private final Map<String, Integer> columns;
        private final Map<String, Trace> traces;
        private Sample next;
        private long row = 1;

        PredictionReplay(Path input, Map<String, Trace> traces) throws IOException {
            this.reader = Files.newBufferedReader(input);
            this.traces = traces;
            this.columns = readColumns(reader.readLine());
            for (Trace trace : traces.values()) trace.latestSample = null;
            this.next = readNext();
        }

        void advanceTo(LocalDateTime time) throws IOException {
            while (next != null && !next.timestamp.isAfter(time)) {
                String id = nextVmId;
                Trace trace = traces.get(id);
                if (trace == null) throw new IOException("Unknown VM in prediction replay: " + id);
                trace.latestSample = next;
                next = readNext();
            }
        }

        private String nextVmId;
        private Sample readNext() throws IOException {
            String line;
            while ((line = reader.readLine()) != null) {
                row++;
                if (line.isBlank()) continue;
                try {
                    List<String> values = parseCsvLine(line);
                    nextVmId = value(values, columns, "vm_id");
                    return sample(values, columns);
                } catch (Exception ex) {
                    throw new IOException("Could not parse CSV row " + row + ": " + ex.getMessage(), ex);
                }
            }
            nextVmId = null;
            return null;
        }

        @Override public void close() throws IOException { reader.close(); }
    }

    private static final class TraceCpu extends UtilizationModelAbstract {
        private final Trace trace;
        private final LocalDateTime start;
        private final PredictionReplay replay;
        TraceCpu(Trace trace, LocalDateTime start, PredictionReplay replay) {
            super(Unit.PERCENTAGE); this.trace = trace; this.start = start; this.replay = replay;
        }
        @Override protected double getUtilizationInternal(double seconds) {
            if (seconds < 0) return 0;
            LocalDateTime at = start.plusSeconds((long)seconds);
            try { replay.advanceTo(at); }
            catch (IOException ex) { throw new UncheckedIOException(ex); }
            Sample sample = trace.latestSample;
            if (sample == null || Duration.between(sample.timestamp, at).getSeconds() > STALE_SECONDS) return 0;
            return Math.min(1.0, sample.currentCpuPct / 100.0);
        }
    }

    private static final class DecisionRow {
        final String timestamp, vmId, decision, sourceHost, targetHost, reason;
        final double probability, threshold;
        DecisionRow(String timestamp, String vmId, double probability, double threshold, String decision,
                    String sourceHost, String targetHost, String reason) {
            this.timestamp = timestamp; this.vmId = vmId; this.probability = probability; this.threshold = threshold;
            this.decision = decision; this.sourceHost = sourceHost; this.targetHost = targetHost; this.reason = reason;
        }
    }

    // Power model constants: idle=100W, peak=250W (SPECpower-inspired linear model)
    private static final double POWER_IDLE_WATTS  = 100.0;
    private static final double POWER_PEAK_WATTS  = 250.0;

    private enum Policy { ML_PROACTIVE, REACTIVE, NO_MIGRATION, MMT, PABFD }

    private static final class PendingMigration {
        final Host source, target;
        final Vm vm;
        PendingMigration(Host source, Host target, Vm vm) { this.source = source; this.target = target; this.vm = vm; }
    }

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUT);
        if (!Files.isRegularFile(INPUT)) throw new FileNotFoundException("Prediction CSV not found: " + INPUT.toAbsolutePath());
        Map<String, Trace> traces = readTraces();
        if (traces.isEmpty()) throw new IllegalStateException("No valid VM traces were loaded.");
        LocalDateTime start = traces.values().stream().map(t -> t.firstTimestamp).min(LocalDateTime::compareTo).orElseThrow();
        LocalDateTime end = traces.values().stream().map(t -> t.lastTimestamp).max(LocalDateTime::compareTo).orElseThrow();
        double duration = Duration.between(start, end).getSeconds();
        System.out.printf(Locale.ROOT, "Loaded %d VMs and %d prediction rows. Timeline: %s to %s.%n",
            traces.size(), traces.values().stream().mapToLong(t -> t.sampleCount).sum(), start, end);
        Map<Policy, long[]> results = new EnumMap<>(Policy.class);
        for (Policy policy : Policy.values()) {
            Path output;
            if (policy == Policy.ML_PROACTIVE) {
                output = OUT;
            } else {
                output = OUT.resolve("comparison").resolve(policy.name().toLowerCase(Locale.ROOT));
            }
            Files.createDirectories(output);
            results.put(policy, runScenario(traces, start, end, duration, policy, output));
        }
        writeComparison(results);
        System.out.println("Simulation scenarios finished. Primary ML outputs: " + OUT.toAbsolutePath());
    }

    private static long[] runScenario(Map<String, Trace> traces, LocalDateTime start, LocalDateTime end,
                                      double duration, Policy policy, Path output) throws IOException {
        CloudSimPlus simulation = new CloudSimPlus();
        List<Host> hosts = createHosts();
        DatacenterSimple datacenter = new DatacenterSimple(simulation, hosts, new VmAllocationPolicyFirstFit());
        datacenter.setSchedulingInterval(300);
        datacenter.enableMigrations();
        DatacenterBrokerSimple broker = new DatacenterBrokerSimple(simulation);
        PredictionReplay replay = new PredictionReplay(INPUT, traces);
        List<Vm> vms = new ArrayList<>();
        List<Cloudlet> cloudlets = new ArrayList<>();
        Map<Vm, Trace> tracesByVm = new IdentityHashMap<>();
        for (Trace t : traces.values()) {
            t.provisionedRamMB = t.peakRamMB();
            Vm vm = new VmSimple(VM_MIPS, VM_PES);
            vm.setRam(t.provisionedRamMB).setBw(VM_BW).setSize(VM_STORAGE_MB);
            t.vm = vm;
            vms.add(vm);
            tracesByVm.put(vm, t);
            Cloudlet cloudlet = new CloudletSimple(CLOUDLET_LENGTH, VM_PES);
            cloudlet.setUtilizationModelCpu(new TraceCpu(t, start, replay));
            cloudlets.add(cloudlet);
        }

        List<String[]> migrationEvents = Collections.synchronizedList(new ArrayList<>());
        List<String[]> hostUtilization = Collections.synchronizedList(new ArrayList<>());
        Map<String, Double> lastMigrationTime = new HashMap<>();
        Set<String> migrationPending = new HashSet<>();
        Map<String, PendingMigration> pendingMigrations = new HashMap<>();
        for (Trace t : traces.values()) {
            Vm vm = t.vm;
            vm.addOnMigrationStartListener(e -> {
                PendingMigration request = pendingMigrations.get(t.sourceId);
                migrationEvents.add(new String[]{fmt(simulation.clock()), t.sourceId,
                        request == null ? hostId(vm.getHost()) : hostId(request.source),
                        request == null ? "" : hostId(request.target), "START", "CloudSim migration started"});
            });
        }
        datacenter.addOnVmMigrationFinishListener((DatacenterVmMigrationEventInfo event) -> {
            Vm vm = event.getVm();
            Trace trace = tracesByVm.get(vm);
            if (trace == null) return;
            PendingMigration request = pendingMigrations.remove(trace.sourceId);
            boolean succeeded = request != null && event.isMigrationSuccessful() && vm.getHost() == request.target;
            migrationEvents.add(new String[]{fmt(event.getTime()), trace.sourceId,
                    request == null ? "" : hostId(request.source),
                    request == null ? "" : hostId(request.target),
                    succeeded ? "SUCCEEDED" : "FAILED",
                    request == null ? "CloudSim reported a migration without a matching policy request"
                        : "cloudsim_success=" + event.isMigrationSuccessful()
                        + "; suitable_ram=" + event.getSuitability().forRam()
                        + "; suitable_pes=" + event.getSuitability().forPes()
                        + "; suitable_bw=" + event.getSuitability().forBw()
                        + "; suitable_storage=" + event.getSuitability().forStorage()
                        + "; actual_host=" + hostId(vm.getHost())});
            migrationPending.remove(trace.sourceId);
        });
        broker.submitVmList(vms);
        broker.submitCloudletList(cloudlets);
        List<DecisionRow> decisions = Collections.synchronizedList(new ArrayList<>());
        final long[] lastDecisionBucket = {-1};
        final int[] initialOccupiedHosts = {-1};
        simulation.addOnClockTickListener((EventInfo event) -> {
            double now = simulation.clock();
            LocalDateTime traceTime = start.plusSeconds((long)now);
            try { replay.advanceTo(traceTime); }
            catch (IOException ex) { throw new UncheckedIOException(ex); }
            if (initialOccupiedHosts[0] < 0) {
                initialOccupiedHosts[0] = (int)hosts.stream().filter(host -> !host.getVmList().isEmpty()).count();
            }
            long bucket = (long)Math.floor(now / DECISION_INTERVAL);
            if (bucket <= 0 || bucket == lastDecisionBucket[0]) return;
            lastDecisionBucket[0] = bucket;
            if (traceTime.isAfter(end.plusSeconds((long)STALE_SECONDS))) return;

            Map<Host, Double> cpuLoad = new HashMap<>(), ramLoad = new HashMap<>();
            Map<Host, Double> currentCpuLoad = new HashMap<>(), currentRamLoad = new HashMap<>();
            Map<Host, Integer> reservedPes = new HashMap<>();
            Map<Host, Long> reservedRamMB = new HashMap<>();
            for (Host host : hosts) {
                cpuLoad.put(host, 0.0); ramLoad.put(host, 0.0);
                currentCpuLoad.put(host, 0.0); currentRamLoad.put(host, 0.0);
                Set<Vm> assigned = Collections.newSetFromMap(new IdentityHashMap<>());
                assigned.addAll(host.getVmList());
                assigned.addAll(host.getVmsMigratingIn());
                for (PendingMigration request : pendingMigrations.values()) {
                    if (request.target == host) assigned.add(request.vm);
                }
                int pes = 0;
                long ramMb = 0;
                for (Vm assignedVm : assigned) {
                    Trace assignedTrace = tracesByVm.get(assignedVm);
                    if (assignedTrace == null) continue;
                    pes += VM_PES;
                    ramMb += assignedTrace.provisionedRamMB;
                }
                reservedPes.put(host, pes);
                reservedRamMB.put(host, ramMb);
            }
            Map<String, Sample> currentSamples = new HashMap<>();
            for (Trace t : traces.values()) {
                if (t.vm.getHost() == Host.NULL) continue;
                Sample sample = t.latestSample;
                if (sample == null || Duration.between(sample.timestamp, traceTime).getSeconds() > STALE_SECONDS) continue;
                currentSamples.put(t.sourceId, sample);
                Host host = t.vm.getHost();
                cpuLoad.merge(host, sample.cpuDemandPct() / 100.0 * VM_PES / HOST_PES, Double::sum);
                ramLoad.merge(host, sample.ramDemandMB(), Double::sum);
                currentCpuLoad.merge(host, sample.currentCpuPct / 100.0 * VM_PES / HOST_PES, Double::sum);
                currentRamLoad.merge(host, sample.currentMemoryKB / 1024.0, Double::sum);
            }
            for (Host host : hosts) {
                if (host.getVmList().isEmpty()) continue;
                hostUtilization.add(new String[]{traceTime.toString(), hostId(host),
                        Double.toString(currentCpuLoad.get(host)), Double.toString(cpuLoad.get(host)),
                        Double.toString(currentRamLoad.get(host)), Double.toString(ramLoad.get(host)),
                        Double.toString(currentRamLoad.get(host) / HOST_RAM_MB),
                        Double.toString(ramLoad.get(host) / HOST_RAM_MB),
                        Integer.toString(host.getVmList().size())});
            }

            Set<Host> hostsReceivingMigration = new HashSet<>();
            Set<Host> hostsInMigration = new HashSet<>();
            for (Map.Entry<String, PendingMigration> entry : pendingMigrations.entrySet()) {
                Trace trace = traces.get(entry.getKey());
                Sample sample = trace.latestSample;
                PendingMigration request = entry.getValue();
                hostsInMigration.add(request.source);
                hostsInMigration.add(request.target);
                if (sample == null || Duration.between(sample.timestamp, traceTime).getSeconds() > STALE_SECONDS) continue;
                hostsReceivingMigration.add(request.target);
                cpuLoad.merge(request.target, sample.cpuDemandPct() / 100.0 * VM_PES / HOST_PES, Double::sum);
                ramLoad.merge(request.target, sample.ramDemandMB(), Double::sum);
            }

            List<Trace> candidates = new ArrayList<>();
            if (policy == Policy.ML_PROACTIVE) {
                // PatchTST risk-alert: only alert VMs, sorted by risk descending
                for (Trace t : traces.values()) {
                    Sample sample = currentSamples.get(t.sourceId);
                    if (sample != null && sample.alert) candidates.add(t);
                }
                candidates.sort((a, b) -> Double.compare(currentSamples.get(b.sourceId).risk, currentSamples.get(a.sourceId).risk));
            } else if (policy == Policy.REACTIVE) {
                // Reactive: all active VMs, sorted by combined current CPU+RAM load descending
                for (Trace t : traces.values()) if (currentSamples.containsKey(t.sourceId)) candidates.add(t);
                candidates.sort((a, b) -> {
                    Sample sa = currentSamples.get(a.sourceId), sb = currentSamples.get(b.sourceId);
                    return Double.compare(sb.currentCpuPct / 100.0 * VM_PES / HOST_PES + sb.currentMemoryKB / HOST_RAM_MB,
                            sa.currentCpuPct / 100.0 * VM_PES / HOST_PES + sa.currentMemoryKB / HOST_RAM_MB);
                });
            } else if (policy == Policy.MMT) {
                // MMT: select VMs from overloaded hosts only; sort by provisioned RAM ascending
                // (smallest RAM = shortest estimated migration time = least data to transfer)
                for (Trace t : traces.values()) {
                    if (!currentSamples.containsKey(t.sourceId)) continue;
                    Host h = t.vm.getHost();
                    if (h == Host.NULL) continue;
                    double hCpu = currentCpuLoad.getOrDefault(h, 0.0);
                    double hRam = currentRamLoad.getOrDefault(h, 0.0);
                    if (hCpu > HOST_CPU_LIMIT || hRam > HOST_RAM_MB * HOST_RAM_LIMIT) candidates.add(t);
                }
                candidates.sort(Comparator.comparingLong(t -> t.provisionedRamMB));
            } else if (policy == Policy.PABFD) {
                // PABFD: all active VMs sorted by current CPU utilisation descending
                for (Trace t : traces.values()) if (currentSamples.containsKey(t.sourceId)) candidates.add(t);
                candidates.sort((a, b) -> Double.compare(
                        currentSamples.get(b.sourceId).currentCpuPct,
                        currentSamples.get(a.sourceId).currentCpuPct));
            }
            Set<Host> sourceHostsMigratingThisInterval = new HashSet<>();
            Set<Host> targetHostsReceivingThisInterval = new HashSet<>();
            for (Trace t : candidates) {
                Sample sample = currentSamples.get(t.sourceId);
                Host source = t.vm.getHost();
                if (source == Host.NULL) {
                    decisions.add(new DecisionRow(traceTime.toString(), t.sourceId, sample.risk, sample.threshold,
                            "UNASSIGNED", "NONE", "", "VM has no CloudSim host assignment"));
                    continue;
                }
                if (migrationPending.contains(t.sourceId)) {
                    decisions.add(new DecisionRow(traceTime.toString(), t.sourceId, sample.risk, sample.threshold,
                            "MIGRATION_PENDING", hostId(source), "", "Prior CloudSim migration has not completed"));
                    continue;
                }
                if (!pendingMigrations.isEmpty()) {
                    decisions.add(new DecisionRow(traceTime.toString(), t.sourceId, sample.risk, sample.threshold,
                            "MIGRATION_DEFERRED", hostId(source), "", "Another CloudSim migration is still in flight"));
                    continue;
                }
                // Determine which load maps and overload criterion to use per policy
                boolean proactive = (policy == Policy.ML_PROACTIVE);
                boolean useForecast = proactive; // MMT/PABFD/REACTIVE all use current load
                Map<Host, Double> decisionCpu = useForecast ? cpuLoad : currentCpuLoad;
                Map<Host, Double> decisionRam = useForecast ? ramLoad : currentRamLoad;
                double sourceCpu = decisionCpu.getOrDefault(source, 0.0);
                double sourceRam = decisionRam.getOrDefault(source, 0.0);
                boolean overloaded = sourceCpu > HOST_CPU_LIMIT || sourceRam > HOST_RAM_MB * HOST_RAM_LIMIT;
                // MMT and PABFD only act on overloaded hosts (candidates already filtered for MMT)
                if ((policy == Policy.MMT || policy == Policy.PABFD) && !overloaded) {
                    decisions.add(new DecisionRow(traceTime.toString(), t.sourceId, sample.risk, sample.threshold,
                            "KEEP", hostId(source), "", "Source host within limits; no migration needed"));
                    continue;
                }
                String decision = "KEEP", targetId = "";
                String reason;
                switch (policy) {
                    case ML_PROACTIVE -> reason = "Risk alert; source host remains within configured forecast limits";
                    case MMT          -> reason = "MMT: source host overloaded; evaluating min-RAM VM migration";
                    case PABFD        -> reason = "PABFD: source host overloaded; evaluating power-minimising placement";
                    default           -> reason = "Current-load baseline; source host remains within configured limits";
                }
                if (now - lastMigrationTime.getOrDefault(t.sourceId, -1e99) < COOLDOWN_SECONDS) {
                    reason = "Migration cooldown active";
                } else if (overloaded) {
                    if (sourceHostsMigratingThisInterval.contains(source) || hostsInMigration.contains(source)) {
                        decisions.add(new DecisionRow(traceTime.toString(), t.sourceId, sample.risk, sample.threshold,
                                "MIGRATION_DEFERRED", hostId(source), "", "One migration already scheduled from this host this interval"));
                        continue;
                    }
                    double candidateCpu = useForecast
                            ? sample.cpuDemandPct() / 100.0 * VM_PES / HOST_PES
                            : sample.currentCpuPct  / 100.0 * VM_PES / HOST_PES;
                    double candidateRam = useForecast ? sample.ramDemandMB() : sample.currentMemoryKB / 1024.0;
                    Host best = null;
                    boolean bestTargetOccupied = false;
                    double bestProjectedLoad   = Double.NEGATIVE_INFINITY;
                    double bestPowerIncrease   = Double.POSITIVE_INFINITY;
                    for (Host target : hosts) {
                        if (target == source || target == Host.NULL) continue;
                        if (targetHostsReceivingThisInterval.contains(target) || hostsInMigration.contains(target)) continue;
                        if (target.hasMigratingVms() || target.getFreePesNumber() < VM_PES) continue;
                        double projectedCpu = decisionCpu.getOrDefault(target, 0.0) + candidateCpu;
                        double projectedRam = decisionRam.getOrDefault(target, 0.0) + candidateRam;
                        if (reservedPes.getOrDefault(target, 0) + VM_PES + MIGRATION_PE_HEADROOM > HOST_PES) continue;
                        if (reservedRamMB.getOrDefault(target, 0L) + t.provisionedRamMB > HOST_RAM_MB) continue;
                        if (target.getFreePesNumber() < VM_PES + MIGRATION_PE_HEADROOM) continue;
                        if (projectedCpu > HOST_CPU_LIMIT || projectedRam > HOST_RAM_MB * HOST_RAM_LIMIT) continue;
                        if (!target.isSuitableForVm(t.vm)) continue;
                        if (policy == Policy.PABFD) {
                            // PABFD: pick host with minimum power increase after placement
                            double currentHostCpu = decisionCpu.getOrDefault(target, 0.0);
                            double powerBefore = POWER_IDLE_WATTS + (POWER_PEAK_WATTS - POWER_IDLE_WATTS) * currentHostCpu;
                            double powerAfter  = POWER_IDLE_WATTS + (POWER_PEAK_WATTS - POWER_IDLE_WATTS) * projectedCpu;
                            double powerIncrease = powerAfter - powerBefore;
                            if (best == null || powerIncrease < bestPowerIncrease) {
                                best = target;
                                bestPowerIncrease = powerIncrease;
                            }
                        } else {
                            // ML_PROACTIVE, REACTIVE, MMT: best-fit (highest projected load that still fits)
                            boolean targetOccupied = !target.getVmList().isEmpty() || hostsReceivingMigration.contains(target);
                            double projectedLoad = Math.max(projectedCpu / HOST_CPU_LIMIT,
                                    projectedRam / (HOST_RAM_MB * HOST_RAM_LIMIT));
                            if (best == null
                                    || (targetOccupied && !bestTargetOccupied)
                                    || (targetOccupied == bestTargetOccupied && projectedLoad > bestProjectedLoad)) {
                                best = target;
                                bestTargetOccupied = targetOccupied;
                                bestProjectedLoad  = projectedLoad;
                            }
                        }
                    }
                    if (best == null) {
                        decision = "NO_FEASIBLE_HOST";
                        reason = "Source host exceeds modeled limit; no destination passes CPU/RAM checks";
                    } else {
                        decision = "MIGRATION_REQUESTED";
                        targetId = String.valueOf(best.getId());
                        reason = switch (policy) {
                            case ML_PROACTIVE -> "Risk alert and projected source-host CPU/RAM overload";
                            case MMT          -> "MMT: min-RAM VM selected from overloaded host";
                            case PABFD        -> "PABFD: power-minimising destination selected";
                            default           -> "Current source-host CPU/RAM overload";
                        };
                        migrationPending.add(t.sourceId);
                        lastMigrationTime.put(t.sourceId, now);
                        pendingMigrations.put(t.sourceId, new PendingMigration(source, best, t.vm));
                        migrationEvents.add(new String[]{fmt(now), t.sourceId, hostId(source), hostId(best), "REQUESTED", reason});
                        datacenter.requestVmMigration(t.vm, best);
                        decisionCpu.merge(best, candidateCpu, Double::sum);
                        decisionRam.merge(best, candidateRam, Double::sum);
                        hostsReceivingMigration.add(best);
                        hostsInMigration.add(source);
                        hostsInMigration.add(best);
                        sourceHostsMigratingThisInterval.add(source);
                        targetHostsReceivingThisInterval.add(best);
                        reservedPes.merge(best, VM_PES, Integer::sum);
                        reservedRamMB.merge(best, t.provisionedRamMB, Long::sum);
                    }
                }
                decisions.add(new DecisionRow(traceTime.toString(), t.sourceId, sample.risk, sample.threshold,
                        decision, hostId(source), targetId, reason));
            }
        });

        simulation.terminateAt(Math.max(1, duration + 1200));
        try { simulation.start(); }
        finally { replay.close(); }
        for (Map.Entry<String, PendingMigration> entry : new ArrayList<>(pendingMigrations.entrySet())) {
            PendingMigration request = entry.getValue();
            boolean reachedTarget = request.vm.getHost() == request.target;
            migrationEvents.add(new String[]{fmt(simulation.clock()), entry.getKey(), hostId(request.source),
                hostId(request.target), reachedTarget ? "SUCCEEDED" : "INCOMPLETE",
                reachedTarget ? "Final CloudSim host assignment matches requested destination"
                    : "Simulation ended before VM reached target; actual_host=" + hostId(request.vm.getHost())});
            pendingMigrations.remove(entry.getKey());
            migrationPending.remove(entry.getKey());
        }
        writeDecisions(decisions, output);
        writeMigrationEvents(migrationEvents, output);
        writeHostUtilization(hostUtilization, output);
        writePlacements(traces, output);
        long[] counts = writeSummary(traces, hosts, decisions, migrationEvents, hostUtilization,
            initialOccupiedHosts[0],
            start, end, policy, output);
        System.out.printf(Locale.ROOT, "%s complete: requests=%d successful=%d failed=%d%n",
                policy, counts[0], counts[1], counts[2]);
        return counts;
    }
    private static Map<String, Trace> readTraces() throws IOException {
        Map<String, Trace> traces = new LinkedHashMap<>();
        try (BufferedReader r = Files.newBufferedReader(INPUT)) {
            Map<String, Integer> col = readColumns(r.readLine());
            String line; long row = 1;
            LocalDateTime previousTimestamp = null;
            while ((line = r.readLine()) != null) {
                row++; if (line.isBlank()) continue;
                try {
                    List<String> values = parseCsvLine(line);
                    String id = value(values, col, "vm_id");
                    Sample sample = sample(values, col);
                    if (previousTimestamp != null && sample.timestamp.isBefore(previousTimestamp)) {
                        throw new IllegalArgumentException("Prediction CSV must be globally chronological");
                    }
                    previousTimestamp = sample.timestamp;
                    traces.computeIfAbsent(id, Trace::new).observe(sample);
                } catch (Exception ex) {
                    throw new IOException("Could not parse CSV row " + row + ": " + ex.getMessage(), ex);
                }
            }
        }
        return traces;
    }

    private static Map<String, Integer> readColumns(String headerLine) throws IOException {
        if (headerLine == null) throw new IOException("CSV is empty.");
        List<String> headers = parseCsvLine(headerLine);
        Map<String, Integer> columns = new HashMap<>();
        for (int i = 0; i < headers.size(); i++) columns.put(headers.get(i).trim(), i);
        for (String name : List.of("vm_id", "decision_time", "risk_probability", "risk_threshold",
                "current_cpu_pct", "current_memory_kb", "forecast_cpu_max", "forecast_memory_max")) {
            if (!columns.containsKey(name)) throw new IOException("Required CSV column missing: " + name);
        }
        return columns;
    }

    private static Sample sample(List<String> values, Map<String, Integer> columns) {
        String timestampText = value(values, columns, "decision_time");
        LocalDateTime timestamp;
        try { timestamp = LocalDateTime.parse(timestampText, FMT); }
        catch (Exception ignored) { timestamp = LocalDateTime.parse(timestampText); }
        return new Sample(timestamp,
                number(values, columns, "current_cpu_pct"),
                number(values, columns, "current_memory_kb"),
                number(values, columns, "forecast_cpu_max"),
                number(values, columns, "forecast_memory_max"),
                number(values, columns, "risk_probability"),
                number(values, columns, "risk_threshold"));
    }

    private static List<String> parseCsvLine(String line) {
        List<String> out = new ArrayList<>(); StringBuilder field = new StringBuilder(); boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (ch == '"') {
                if (quoted && i + 1 < line.length() && line.charAt(i + 1) == '"') { field.append('"'); i++; }
                else quoted = !quoted;
            } else if (ch == ',' && !quoted) { out.add(field.toString()); field.setLength(0); }
            else field.append(ch);
        }
        out.add(field.toString()); return out;
    }
    private static String value(List<String> v, Map<String,Integer> c, String key) {
        int i = c.get(key); if (i >= v.size()) throw new IllegalArgumentException("Missing value for " + key);
        return v.get(i).trim();
    }
    private static double number(List<String> v, Map<String,Integer> c, String key) {
        double d = Double.parseDouble(value(v,c,key));
        if (!Double.isFinite(d)) throw new IllegalArgumentException("Non-finite value for " + key);
        return d;
    }
    private static List<Host> createHosts() {
        List<Host> hosts = new ArrayList<>();
        for (int i = 0; i < HOST_COUNT; i++) {
            List<Pe> pes = new ArrayList<>();
            for (int p = 0; p < HOST_PES; p++) pes.add(new PeSimple(HOST_MIPS));
            HostSimple h = new HostSimple(HOST_RAM_MB, HOST_BW, HOST_STORAGE_MB, pes);
            h.setRamProvisioner(new ResourceProvisionerSimple());
            h.setBwProvisioner(new ResourceProvisionerSimple());
            // Linear power model: idle=100W at 0% CPU, peak=250W at 100% CPU
            h.setPowerModel(new PowerModelHostSimple(POWER_PEAK_WATTS, POWER_IDLE_WATTS));
            h.enableUtilizationStats();
            hosts.add(h);
        }
        return hosts;
    }
    private static void writeDecisions(List<DecisionRow> rows, Path output) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(output.resolve("decision_plan.csv"))) {
            w.write("timestamp,vm_id,risk_probability,risk_threshold,decision,current_host,proposed_host,reason\n");
            synchronized (rows) { for (DecisionRow r : rows) w.write(csv(r.timestamp)+","+csv(r.vmId)+","+r.probability+","+r.threshold+","+csv(r.decision)+","+csv(r.sourceHost)+","+csv(r.targetHost)+","+csv(r.reason)+"\n"); }
        }
    }
    private static void writeMigrationEvents(List<String[]> rows, Path output) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(output.resolve("migration_events.csv"))) {
            w.write("simulation_time,source_vm_id,source_host,target_host,event,details\n");
            synchronized (rows) {
                for (String[] r : rows) {
                    w.write(csv(r[0])+","+csv(r[1])+","+csv(r[2])+","+csv(r[3])+","+csv(r[4])+","+csv(r[5])+"\n");
                }
            }
        }
    }
    private static void writeHostUtilization(List<String[]> rows, Path output) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(output.resolve("host_utilization.csv"))) {
            w.write("timestamp,host_id,current_cpu_fraction,forecast_cpu_fraction,current_ram_mb,forecast_ram_mb,current_ram_fraction,forecast_ram_fraction,vm_count\n");
            synchronized (rows) {
                for (String[] r : rows) {
                    w.write(csv(r[0])+","+csv(r[1])+","+r[2]+","+r[3]+","+r[4]+","+r[5]+","+r[6]+","+r[7]+","+r[8]+"\n");
                }
            }
        }
    }
    private static void writePlacements(Map<String, Trace> traces, Path output) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(output.resolve("final_vm_placement.csv"))) {
            w.write("source_vm_id,simulated_vm_id,final_host,provisioned_ram_mb,sample_count,risk_alert_samples\n");
            for (Trace t : traces.values()) {
                w.write(csv(t.sourceId)+","+t.vm.getId()+","+csv(hostId(t.vm.getHost()))+","+t.provisionedRamMB+","+t.sampleCount+","+t.alertCount+"\n");
            }
        }
    }
    /**
     * Compute total energy in kWh from host power models over the simulation.
     * Uses the linear model: P(u) = idle + (peak-idle)*u integrated over uptime.
     * getCpuPercentUtilization() returns the time-averaged CPU fraction for the host.
     */
    private static double computeEnergyKwh(List<Host> hosts) {
        double totalWattSeconds = 0;
        for (Host h : hosts) {
            double uptimeSec = h.getTotalUpTime();
            if (uptimeSec <= 0) continue;
            double avgCpu = h.getCpuUtilizationStats().getMean();
            if (!Double.isFinite(avgCpu) || avgCpu < 0) avgCpu = 0;
            double avgWatts = POWER_IDLE_WATTS + (POWER_PEAK_WATTS - POWER_IDLE_WATTS) * avgCpu;
            totalWattSeconds += avgWatts * uptimeSec;
        }
        return totalWattSeconds / 3_600_000.0;
    }

    private static long[] writeSummary(Map<String, Trace> traces, List<Host> hosts, List<DecisionRow> decisions,
                                     List<String[]> events, List<String[]> hostUtilization,
                                     int initialOccupiedHosts,
                                     LocalDateTime start, LocalDateTime end,
                                     Policy policy, Path output) throws IOException {
        long rows = traces.values().stream().mapToLong(t -> t.sampleCount).sum();
        long alerts = traces.values().stream().mapToLong(t -> t.alertCount).sum();
        long allocated = traces.values().stream().filter(t -> t.vm.getHost() != Host.NULL).count();
        long occupiedAtEnd = traces.values().stream().map(t -> t.vm.getHost())
            .filter(host -> host != null && host != Host.NULL).map(Host::getId).distinct().count();
        long requested = decisions.stream().filter(d -> d.decision.equals("MIGRATION_REQUESTED")).count();
        long noFeasible = decisions.stream().filter(d -> d.decision.equals("NO_FEASIBLE_HOST")).count();
        long starts = events.stream().filter(e -> e[4].equals("START")).count();
        long succeeded = events.stream().filter(e -> e[4].equals("SUCCEEDED")).count();
        long failed = events.stream().filter(e -> e[4].equals("FAILED")).count();
        long incomplete = events.stream().filter(e -> e[4].equals("INCOMPLETE")).count();
        double maxForecastCpu = hostUtilization.stream().mapToDouble(r -> Double.parseDouble(r[3])).max().orElse(0);
        double maxForecastRam = hostUtilization.stream().mapToDouble(r -> Double.parseDouble(r[5])).max().orElse(0);
        long cpuOverloadSnapshots = hostUtilization.stream().filter(r -> Double.parseDouble(r[3]) > HOST_CPU_LIMIT).count();
        long ramOverloadSnapshots = hostUtilization.stream().filter(r -> Double.parseDouble(r[5]) > HOST_RAM_MB * HOST_RAM_LIMIT).count();
        try (BufferedWriter w = Files.newBufferedWriter(output.resolve("simulation_summary.csv"))) {
            w.write("metric,value,notes\n");
            summaryRow(w,"policy",policy,"Policy scenario");
            summaryRow(w,"host_ram_mb",HOST_RAM_MB,"Assumed host capacity; not a Bitbrains fact");
            summaryRow(w,"host_pes",HOST_PES,"Assumed host CPU capacity; not a Bitbrains fact");
            summaryRow(w,"cpu_limit_fraction",HOST_CPU_LIMIT,"Configured forecast/current CPU threshold");
            summaryRow(w,"ram_limit_fraction",HOST_RAM_LIMIT,"Configured forecast/current RAM threshold");
            summaryRow(w,"vm_vcpus",VM_PES,"Assumed VM CPU allocation");
            summaryRow(w,"migration_pe_headroom",MIGRATION_PE_HEADROOM,"Additional destination PEs kept free to reduce allocation failures");
            summaryRow(w,"cpu_mapping","vm_cpu_pct * vm_vcpus / host_pes","Linear distribution across assumed VM vCPUs");
            summaryRow(w,"decision_interval_seconds",DECISION_INTERVAL,"Policy evaluation interval");
            summaryRow(w,"migration_cooldown_seconds",COOLDOWN_SECONDS,"Per-VM migration cooldown");
            summaryRow(w,"stale_sample_seconds",STALE_SECONDS,"Older workload observations are ignored");
            summaryRow(w,"prediction_rows",rows,"Rows loaded from ML prediction CSV");
            summaryRow(w,"unique_vms",traces.size(),"Source VM count");
            summaryRow(w,"configured_hosts",hosts.size(),"Assumed host count, not a Bitbrains fact");
            summaryRow(w,"initial_occupied_hosts",initialOccupiedHosts,"Hosts with at least one VM after initial CloudSim allocation");
            summaryRow(w,"occupied_hosts_at_end",occupiedAtEnd,"Hosts with at least one VM; no host power-down model is enabled");
            summaryRow(w,"start_timestamp",start,"Earliest prediction timestamp");
            summaryRow(w,"end_timestamp",end,"Latest prediction timestamp");
            summaryRow(w,"risk_alert_rows",alerts,"risk_probability >= risk_threshold");
            summaryRow(w,"max_forecast_host_cpu_fraction",maxForecastCpu,"Maximum modeled host forecast CPU load");
            summaryRow(w,"max_forecast_host_ram_mb",maxForecastRam,"Maximum modeled host forecast RAM demand");
            summaryRow(w,"host_cpu_overload_snapshots",cpuOverloadSnapshots,"Occupied host snapshots above configured CPU threshold");
            summaryRow(w,"host_ram_overload_snapshots",ramOverloadSnapshots,"Occupied host snapshots above configured RAM threshold");
            String decisionNote = switch (policy) {
                case ML_PROACTIVE -> "Risk-alert candidates evaluated while VM was active and assigned";
                case REACTIVE     -> "Current-load candidates evaluated while VM was active and assigned";
                case MMT          -> "Min-migration-time candidates from overloaded hosts";
                case PABFD        -> "Power-aware best-fit decreasing candidates from overloaded hosts";
                case NO_MIGRATION -> "No-migration baseline does not evaluate move candidates";
            };
            summaryRow(w,"decision_rows",decisions.size(),decisionNote);
            summaryRow(w,"migration_requests",requested,"Requests issued to CloudSim; not necessarily completed");
            summaryRow(w,"no_feasible_destination_decisions",noFeasible,"No target passed modeled CPU/RAM checks");
            summaryRow(w,"migration_start_events",starts,"Observed CloudSim migration start listener events");
            summaryRow(w,"successful_migrations",succeeded,"VM was observed on the requested destination after migration finished");
            summaryRow(w,"failed_migrations",failed,"CloudSim migration finish event did not land on the requested destination");
            summaryRow(w,"incomplete_migrations",incomplete,"No destination arrival was observed before simulation termination");
            summaryRow(w,"allocated_vms_at_end",allocated,"VMs assigned to a host at simulation end");
            double energyKwh = computeEnergyKwh(hosts);
            summaryRow(w,"energy_kwh", String.format(Locale.ROOT, "%.4f", energyKwh),
                    "Linear power model: idle=" + POWER_IDLE_WATTS + "W peak=" + POWER_PEAK_WATTS + "W; integrated over host uptime");
            summaryRow(w,"power_model","PowerModelHostSimple(peak=" + POWER_PEAK_WATTS + ",idle=" + POWER_IDLE_WATTS + ")","SPECpower-inspired linear model");
            summaryRow(w,"sla_violations","NOT_MEASURED","No explicit SLA measurement implemented");
            summaryRow(w,"disk_io_constraints","NOT_MODELED","Disk throughput is not enforced by this CPU/RAM placement policy");
        }
        return new long[]{requested, succeeded, failed + incomplete, noFeasible, decisions.size()};
    }
    private static void writeComparison(Map<Policy, long[]> results) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(OUT.resolve("policy_comparison.csv"))) {
            w.write("policy,migration_requests,successful_migrations,failed_or_incomplete_migrations,no_feasible_destination_decisions,decision_rows\n");
            for (Policy policy : Policy.values()) {
                long[] counts = results.get(policy);
                w.write(policy.name()+","+counts[0]+","+counts[1]+","+counts[2]+","+counts[3]+","+counts[4]+"\n");
            }
        }
    }
    private static void summaryRow(BufferedWriter w, String key, Object val, String note) throws IOException {
        w.write(csv(key)+","+csv(String.valueOf(val))+","+csv(note)+"\n");
    }
    private static String hostId(Host host) { return host == null || host == Host.NULL ? "NONE" : String.valueOf(host.getId()); }
    private static String fmt(double seconds) { return String.format(Locale.ROOT,"%.0f",seconds); }
    private static String csv(String s) { return "\"" + String.valueOf(s).replace("\"","\"\"") + "\""; }
    private static double clamp(double x, double lo, double hi) { return Math.max(lo, Math.min(hi, x)); }
}
