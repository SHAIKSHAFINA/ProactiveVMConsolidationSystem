import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Background, Controls, ReactFlow, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

const API_BASE =
  (import.meta as ImportMeta & { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ??
  'http://127.0.0.1:8000';
const API = `${API_BASE.replace(/\/$/, '')}/api`;
const POLICIES = ['ML_PROACTIVE', 'REACTIVE', 'NO_MIGRATION', 'MMT', 'PABFD'];
const EVENTS = ['REQUESTED', 'START', 'SUCCEEDED', 'FAILED', 'INCOMPLETE'];

type PageKey =
  | 'overview'
  | 'prediction'
  | 'explorer'
  | 'simulation'
  | 'benchmark'
  | 'migrations'
  | 'dataset'
  | 'results-prediction';
type SectionKey = 'pipeline' | 'results';
type GenericRow = Record<string, string | number | boolean | null | undefined>;
type DatasetInfo = {
  dataset_id: string;
  filename: string;
  source: 'demo' | 'upload' | 'local';
  created_at?: string;
  size_bytes: number;
  row_count: number;
  vm_count: number;
  vm_ids?: string[];
  columns: string[];
  validation: {
    valid: boolean;
    errors: string[];
    warnings: string[];
    missing_values: Record<string, number>;
    row_count?: number;
    vm_count?: number;
    columns?: string[];
    optional_columns_present?: string[];
    simulator_compatible?: boolean;
    simulator_errors?: string[];
  };
  preview: GenericRow[];
  dataset_kind?: 'prediction_csv' | 'raw_workload_csv' | 'incompatible_raw_workload_csv' | 'raw_or_incompatible_csv';
};
type CsvValidation = {
  filename: string;
  size_bytes: number;
  validation: DatasetInfo['validation'] & {
    row_count: number;
    vm_count: number;
    columns: string[];
  };
  content_sha256: string;
  duplicate_dataset_id: string | null;
  preview: GenericRow[];
  dataset_kind: 'prediction_csv' | 'raw_workload_csv' | 'incompatible_raw_workload_csv' | 'raw_or_incompatible_csv';
  format_note: string;
  inference_available: boolean;
  dataset_id?: string;
};
type ConfirmationDialog = {
  title: string;
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
};
type ComparisonRow = {
  policy: string;
  run_id?: string;
  migration_requests?: string | number | null;
  successful_migrations?: string | number | null;
  failed_or_incomplete_migrations?: string | number | null;
  no_feasible_destination_decisions?: string | number | null;
  decision_rows?: string | number | null;
  energy_kwh?: string | number | null;
  host_overload_snapshots?: string | number | null;
  host_cpu_overload_snapshots?: string | number | null;
  host_ram_overload_snapshots?: string | number | null;
};
type VmRow = GenericRow & { source_vm_id: string; final_host?: string | null };
type MigrationEvent = GenericRow & {
  simulation_time: string;
  source_vm_id: string;
  source_host: string;
  target_host: string;
  event: string;
  details: string;
};
type DecisionRecord = GenericRow & {
  timestamp: string;
  vm_id: string;
  risk_probability: string;
  risk_threshold: string;
  decision: string;
  current_host: string;
  proposed_host: string;
  reason: string;
};
type HostRow = GenericRow & {
  host_id: string;
  current_cpu_fraction?: number;
  forecast_cpu_fraction?: number;
  current_ram_fraction?: number;
  forecast_ram_fraction?: number;
  vm_count_at_snapshot?: number;
};
type RunStatus = {
  run_id: string;
  policy: string;
  dataset_id: string;
  data_mode?: 'prediction_csv' | 'raw_workload_csv';
  status: string;
  stage: string;
  progress?: number | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  exit_code?: number | null;
  selected_prediction_rows?: number;
  selected_prediction_vms?: number;
  generated_prediction_rows?: number;
  generated_prediction_vms?: number;
  inference_completed_vm_count?: number;
  inference_total_vm_count?: number;
  inference_processed_vm_count?: number;
  inference_cached_vm_count?: number;
  inference_generated_rows?: number;
  inference_mode?: 'fresh_predictions' | 'fresh_and_resumed_predictions' | 'resumed_predictions';
  run_parameters?: {
    vm_ids: string[] | null;
    start_time: string | null;
    end_time: string | null;
    risk_threshold: number;
    host_count: number;
    host_cpu_limit: number;
    host_ram_limit: number;
    host_ram_mb: number;
    host_pes: number;
    compared_policies: string[];
  };
};
type VmDetail = {
  vm_id: string;
  placement: GenericRow | null;
  migrations: MigrationEvent[];
  decisions: DecisionRecord[];
  prediction_series: GenericRow[];
  disk_forecasts_available: boolean;
  prediction_series_note: string;
  disk_constraint_note: string;
  placement_note: string;
  prediction_data_mode?: 'prepared_prediction_csv' | 'fresh_model_inference';
};
type PolicySummary = Record<string, string>;
type Overview = {
  data_source: string;
  dataset_id: string;
  run_id: string | null;
  primary_policy: string;
  primary_summary: PolicySummary;
  policy_comparison: ComparisonRow[];
  supported_policies: string[];
  metric_definitions: Record<string, string>;
};

function parseRunVmSelection(value: string): { vmIds?: string[]; error?: string } {
  const trimmed = value.trim();
  if (!trimmed || trimmed.toLocaleUpperCase() === 'ALL') return {};
  const vmIds = trimmed.split(',').map((vmId) => vmId.trim());
  if (vmIds.some((vmId) => !vmId)) {
    return { error: 'Enter VM IDs separated by commas without empty entries.' };
  }
  if (vmIds.some((vmId) => vmId.toLocaleUpperCase() === 'ALL')) {
    return { error: 'Use ALL by itself, or enter one or more VM IDs.' };
  }
  if (new Set(vmIds).size !== vmIds.length) {
    return { error: 'The VM selection contains duplicate IDs.' };
  }
  if (vmIds.length > 100) {
    return { error: 'Select at most 100 VM IDs per run, or use ALL.' };
  }
  return { vmIds };
}

const NAV: { key: SectionKey; page: PageKey; label: string }[] = [
  { key: 'pipeline', page: 'dataset', label: 'Pipeline' },
  { key: 'results', page: 'overview', label: 'Results' },
];

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  const contentType = response.headers.get('content-type') ?? '';
  const body = contentType.includes('application/json')
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = typeof body === 'object' && body !== null ? body.detail ?? body : body;
    if (typeof detail === 'object' && detail !== null) {
      const message = typeof detail.message === 'string' ? detail.message : '';
      const validationMessage = typeof detail.validation_message === 'string'
        ? detail.validation_message
        : '';
      const formatNote = typeof detail.format_note === 'string' ? detail.format_note : '';
      const validationErrors = Array.isArray(detail.errors)
        ? detail.errors.filter((item: unknown): item is string => typeof item === 'string')
        : [];
      const combined = [message || formatNote, validationMessage, ...validationErrors]
        .filter(Boolean)
        .join(' ');
      throw new Error(combined || JSON.stringify(detail));
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return body as T;
}

async function fetchAllPages<T>(
  fetchPage: (page: number, pageSize: number) => Promise<{ total: number; records: T[] }>,
  pageSize = 1000,
): Promise<T[]> {
  const records: T[] = [];
  for (let page = 1; ; page += 1) {
    const result = await fetchPage(page, pageSize);
    records.push(...result.records);
    if (records.length >= result.total) return records;
    if (result.records.length === 0) {
      throw new Error(`Paged API returned no records before reaching its total of ${result.total}.`);
    }
  }
}

function query(values: Record<string, string | number | undefined | null>): string {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
  });
  const result = params.toString();
  return result ? `?${result}` : '';
}

function numeric(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function display(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === '') return '—';
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? parsed.toLocaleString(undefined, { maximumFractionDigits: digits })
    : String(value);
}

function eventClass(event: string): string {
  if (event === 'SUCCEEDED') return 'badge success';
  if (event === 'FAILED' || event === 'INCOMPLETE') return 'badge danger';
  if (event === 'START') return 'badge info';
  return 'badge warning';
}

function MetricCard({ label, value, detail, tone = 'blue' }: {
  label: string;
  value: string;
  detail?: string;
  tone?: 'blue' | 'green' | 'amber' | 'red';
}) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </article>
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <div className="empty-mark">—</div>
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function ConfirmationDialogView({
  confirmation,
  busy,
  onCancel,
  onConfirm,
}: {
  confirmation: ConfirmationDialog;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel();
      }}
    >
      <section className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <h2 id="confirm-title">{confirmation.title}</h2>
        <p>{confirmation.message}</p>
        <div className="button-row">
          <button className="button secondary" onClick={onCancel}>Cancel</button>
          <button className="button primary" disabled={busy} onClick={onConfirm}>
            {confirmation.confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

function LoadingRows({ count = 4 }: { count?: number }) {
  return (
    <div className="skeleton-list" aria-label="Loading">
      {Array.from({ length: count }, (_, index) => <div className="skeleton-row" key={index} />)}
    </div>
  );
}

function DataTable({ columns, rows, emptyTitle = 'No records found', emptyDetail = 'Try changing the search or filter.', paginate = true, resetKey }: {
  columns: { key: string; label: string; render?: (row: GenericRow) => React.ReactNode }[];
  rows: GenericRow[];
  emptyTitle?: string;
  emptyDetail?: string;
  paginate?: boolean;
  resetKey?: string;
}) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  useEffect(() => setPage(1), [resetKey]);
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const visibleRows = paginate
    ? rows.slice((currentPage - 1) * pageSize, currentPage * pageSize)
    : rows;
  if (!rows.length) return <EmptyState title={emptyTitle} detail={emptyDetail} />;
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead><tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr></thead>
          <tbody>
            {visibleRows.map((row, index) => (
              <tr key={`${String(row.id ?? row.vm_id ?? row.source_vm_id ?? row.timestamp ?? '')}-${index}`}>
                {columns.map((column) => (
                  <td key={column.key}>{column.render ? column.render(row) : display(row[column.key], 3)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {paginate && rows.length > pageSize && (
        <Pagination
          page={currentPage}
          total={rows.length}
          pageSize={pageSize}
          onPage={setPage}
          onPageSize={(size) => { setPageSize(size); setPage(1); }}
        />
      )}
    </>
  );
}

function Pagination({ page, total, pageSize, onPage, onPageSize }: {
  page: number;
  total: number;
  pageSize: number;
  onPage: (page: number) => void;
  onPageSize: (pageSize: number) => void;
}) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const currentPage = total === 0 ? 0 : Math.min(Math.max(1, page), pageCount);
  const firstPage = Math.max(1, Math.min(currentPage - 2, pageCount - 4));
  const pageNumbers = total
    ? Array.from(
        { length: Math.min(5, pageCount) },
        (_, index) => firstPage + index,
      )
    : [];
  return (
    <div className="pagination">
      <span>{total.toLocaleString()} records · page {currentPage} of {total ? pageCount : 0}</span>
      <div className="button-row">
        <label className="control-group"><span>Rows</span><select value={pageSize} onChange={(event) => onPageSize(Number(event.target.value))}><option value={10}>10</option><option value={25}>25</option><option value={50}>50</option><option value={100}>100</option></select></label>
        <button className="button secondary small" disabled={currentPage <= 1} onClick={() => onPage(1)}>First</button>
        <button className="button secondary small" disabled={currentPage <= 1} onClick={() => onPage(currentPage - 1)}>Previous</button>
        {pageNumbers.map((number) => (
          <button
            key={number}
            className={`button ${number === currentPage ? 'primary' : 'secondary'} small`}
            aria-current={number === currentPage ? 'page' : undefined}
            onClick={() => onPage(number)}
          >
            {number}
          </button>
        ))}
        <button className="button secondary small" disabled={currentPage >= pageCount} onClick={() => onPage(currentPage + 1)}>Next</button>
        <button className="button secondary small" disabled={currentPage >= pageCount} onClick={() => onPage(pageCount)}>Last</button>
      </div>
    </div>
  );
}

function App() {
  const [page, setPage] = useState<PageKey>('dataset');
  const [overview, setOverview] = useState<Overview | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<{
    active_dataset_id: string;
    active_dataset: DatasetInfo;
    datasets: DatasetInfo[];
    upload_max_bytes: number;
    inference_note: string;
    model_inference_available?: boolean;
    simulation_note: string;
  } | null>(null);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [selectedPolicy, setSelectedPolicy] = useState('ML_PROACTIVE');
  const [policySummary, setPolicySummary] = useState<PolicySummary>({});
  const [comparison, setComparison] = useState<ComparisonRow[]>([]);
  const [vms, setVms] = useState<VmRow[]>([]);
  const [vmTotal, setVmTotal] = useState(0);
  const [vmSearch, setVmSearch] = useState('');
  const [selectedVmId, setSelectedVmId] = useState('');
  const [vmDetail, setVmDetail] = useState<VmDetail | null>(null);
  const [hosts, setHosts] = useState<HostRow[]>([]);
  const [hostSnapshots, setHostSnapshots] = useState<HostRow[]>([]);
  const [placements, setPlacements] = useState<VmRow[]>([]);
  const [selectedHostId, setSelectedHostId] = useState('');
  const [hostSearch, setHostSearch] = useState('');
  const [replayEventKey, setReplayEventKey] = useState('');
  const [migrationEvents, setMigrationEvents] = useState<MigrationEvent[]>([]);
  const [migrationTotal, setMigrationTotal] = useState(0);
  const [hostMigrationEvents, setHostMigrationEvents] = useState<MigrationEvent[]>([]);
  const [migrationSearch, setMigrationSearch] = useState('');
  const [migrationHostFilter, setMigrationHostFilter] = useState('');
  const [eventFilter, setEventFilter] = useState('');
  const [migrationPage, setMigrationPage] = useState(1);
  const [migrationPageSize, setMigrationPageSize] = useState(50);
  const [migrationSortBy, setMigrationSortBy] = useState('simulation_time');
  const [migrationSortOrder, setMigrationSortOrder] = useState<'asc' | 'desc'>('asc');
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [decisionTotal, setDecisionTotal] = useState(0);
  const [decisionSearch, setDecisionSearch] = useState('');
  const [decisionHostFilter, setDecisionHostFilter] = useState('');
  const [decisionFilter, setDecisionFilter] = useState('ACTIONABLE');
  const [decisionPage, setDecisionPage] = useState(1);
  const [decisionPageSize, setDecisionPageSize] = useState(50);
  const [decisionSortBy, setDecisionSortBy] = useState('timestamp');
  const [decisionSortOrder, setDecisionSortOrder] = useState<'asc' | 'desc'>('asc');
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [completedResultRunId, setCompletedResultRunId] = useState('');
  const [runLogs, setRunLogs] = useState<{ stdout: string; stderr: string }>({ stdout: '', stderr: '' });
  const [runPolicy, setRunPolicy] = useState('ML_PROACTIVE');
  const [runVmId, setRunVmId] = useState('ALL');
  const [runStartTime, setRunStartTime] = useState('');
  const [runEndTime, setRunEndTime] = useState('');
  const [runHostCount, setRunHostCount] = useState('200');
  const [runHostRamMb, setRunHostRamMb] = useState('262144');
  const [runHostPes, setRunHostPes] = useState('32');
  const [runCpuLimit, setRunCpuLimit] = useState('0.8');
  const [runRamLimit, setRunRamLimit] = useState('0.8');
  const [runRiskThreshold, setRunRiskThreshold] = useState('0.66');
  const [comparisonMetric, setComparisonMetric] = useState<keyof ComparisonRow>('host_overload_snapshots');
  const [sortMetric, setSortMetric] = useState<keyof ComparisonRow>('energy_kwh');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [benchmarkSearch, setBenchmarkSearch] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<DatasetInfo | null>(null);
  const [uploadValidation, setUploadValidation] = useState<CsvValidation | null>(null);
  const [savedDatasetId, setSavedDatasetId] = useState('');
  const [inferenceNote, setInferenceNote] = useState('');
  const [predictionStart, setPredictionStart] = useState('');
  const [predictionEnd, setPredictionEnd] = useState('');
  const [predictionSearch, setPredictionSearch] = useState('');
  const [health, setHealth] = useState<'checking' | 'online' | 'offline'>('checking');
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [inferenceBusy, setInferenceBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [confirmation, setConfirmation] = useState<ConfirmationDialog | null>(null);
  const policyRequestRef = useRef(0);
  const overviewRequestRef = useRef(0);
  const vmRequestRef = useRef(0);

  useEffect(() => {
    if (!confirmation) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setConfirmation(null);
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [confirmation]);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId) ?? null;
  const activeDataset = datasetInfo?.datasets.find((dataset) => dataset.dataset_id === activeDatasetId);
  const resultRun = runs.find((run) => run.run_id === completedResultRunId) ?? null;
  const policyRunId =
    resultRun?.status === 'COMPLETED' &&
    resultRun.dataset_id === activeDatasetId
      ? resultRun.run_id
      : '';
  const hasDatasetResults = Boolean(policyRunId);
  const hasMatchingResults = Boolean(policyRunId);
  const resultSourceLabel = policyRunId
    ? `${resultRun?.data_mode === 'raw_workload_csv' ? 'Fresh model inference' : 'Prepared prediction inputs'} · Run ${policyRunId}`
    : 'No results yet';

  const loadInitial = useCallback(async () => {
    setLoading(true);
    setError('');
    const [healthResult] = await Promise.allSettled([
      apiRequest<{ status: string; model_inference_note?: string }>('/health'),
    ]);
    if (healthResult.status === 'fulfilled') {
      setInferenceNote(healthResult.value.model_inference_note ?? '');
    }
    setHealth(healthResult.status === 'fulfilled' ? 'online' : 'offline');
    setLoading(false);
  }, []);

  const loadVms = useCallback(async () => {
    const requestId = ++vmRequestRef.current;
    if (!activeDatasetId) {
      setVms([]);
      setVmTotal(0);
      setSelectedVmId('');
      return;
    }
    try {
      const loadedVms = await fetchAllPages(async (page, pageSize) => {
        const response = await apiRequest<{ total: number; vms: VmRow[] }>(
          `/vms${query({ dataset_id: activeDatasetId, run_id: policyRunId || undefined, policy: selectedPolicy, search: vmSearch, page, page_size: pageSize })}`,
        );
        return { total: response.total, records: response.vms };
      }, 2000);
      if (requestId !== vmRequestRef.current) return;
      setVms(loadedVms);
      setVmTotal(loadedVms.length);
      setSelectedVmId((current) =>
        loadedVms.some((vm) => vm.source_vm_id === current)
          ? current
          : loadedVms[0]?.source_vm_id ?? '',
      );
    } catch (requestError) {
      if (requestId !== vmRequestRef.current) return;
      setVms([]);
      setVmTotal(0);
      setError(`Could not load VMs: ${(requestError as Error).message}`);
    }
  }, [activeDatasetId, policyRunId, selectedPolicy, vmSearch]);

  const runModelInference = useCallback(async () => {
    if (
      !activeDatasetId
      || activeDataset?.dataset_kind !== 'raw_workload_csv'
      || !datasetInfo?.model_inference_available
    ) {
      return;
    }
    setInferenceBusy(true);
    setError('');
    setSuccess('');
    try {
      const result = await apiRequest<{
        row_count: number;
        vm_count: number;
        total_prediction_rows: number;
        completed_vm_count: number;
      }>('/dataset/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: activeDatasetId,
          vm_ids: selectedVmId ? [selectedVmId] : undefined,
          start_time: predictionStart || undefined,
          end_time: predictionEnd || undefined,
        }),
      });
      await loadVms();
      setSuccess(
        `Fresh inference added ${result.row_count.toLocaleString()} forecasts `
        + `for ${result.vm_count.toLocaleString()} VM(s). Resumable output now has `
        + `${result.total_prediction_rows.toLocaleString()} predictions across `
        + `${result.completed_vm_count.toLocaleString()} VMs.`,
      );
    } catch (requestError) {
      setError(`Model inference failed: ${(requestError as Error).message}`);
    } finally {
      setInferenceBusy(false);
    }
  }, [
    activeDataset,
    activeDatasetId,
    datasetInfo?.model_inference_available,
    loadVms,
    predictionEnd,
    predictionStart,
    selectedVmId,
  ]);

  const loadPolicyData = useCallback(async () => {
    const requestId = ++policyRequestRef.current;
    if (!activeDatasetId || !hasMatchingResults || !policyRunId) {
      setPolicySummary({});
      setMigrationEvents([]);
      setMigrationTotal(0);
      setDecisions([]);
      setDecisionTotal(0);
      setHosts([]);
      setHostSnapshots([]);
      setHostMigrationEvents([]);
      setPlacements([]);
      setReplayEventKey('');
      return;
    }
    const runQuery = {
      dataset_id: activeDatasetId,
      ...(policyRunId ? { run_id: policyRunId } : {}),
    };
    const params = query(runQuery);
    const outcomes = await Promise.allSettled([
      apiRequest<{ summary: PolicySummary }>(`/policies/${selectedPolicy}/summary${params}`),
      apiRequest<{ total: number; records: MigrationEvent[] }>(
        `/policies/${selectedPolicy}/migrations${query({ ...runQuery, page: migrationPage, page_size: migrationPageSize, event_filter: eventFilter, vm_filter: migrationSearch, host_filter: migrationHostFilter, sort_by: migrationSortBy, sort_order: migrationSortOrder })}`,
      ),
      apiRequest<{ total: number; records: DecisionRecord[] }>(
        `/policies/${selectedPolicy}/decisions${query({ ...runQuery, page: decisionPage, page_size: decisionPageSize, decision_filter: decisionFilter, vm_filter: decisionSearch, host_filter: decisionHostFilter, sort_by: decisionSortBy, sort_order: decisionSortOrder })}`,
      ),
      apiRequest<{ hosts: HostRow[]; snapshots: HostRow[]; host_count: number }>(
        `/policies/${selectedPolicy}/hosts${query(runQuery)}`,
      ),
      fetchAllPages(
        (page, pageSize) => apiRequest<{ total: number; records: VmRow[] }>(
          `/policies/${selectedPolicy}/placements${query({ ...runQuery, page, page_size: pageSize })}`,
        ),
        2000,
      ),
    ]);
    if (requestId !== policyRequestRef.current) return;
    const summaryResult = outcomes[0];
    if (summaryResult.status === 'fulfilled') setPolicySummary(summaryResult.value.summary);
    else {
      setPolicySummary({});
      setError(`Could not load ${selectedPolicy} summary: ${summaryResult.reason.message}`);
    }
    const migrationResult = outcomes[1];
    if (migrationResult.status === 'fulfilled') {
      setMigrationEvents(migrationResult.value.records);
      setMigrationTotal(migrationResult.value.total);
      setMigrationPage((current) => Math.min(
        current,
        Math.max(1, Math.ceil(migrationResult.value.total / migrationPageSize)),
      ));
    } else {
      setMigrationEvents([]);
      setMigrationTotal(0);
      setError((current) => current || `Could not load migration events: ${migrationResult.reason.message}`);
    }
    const decisionResult = outcomes[2];
    if (decisionResult.status === 'fulfilled') {
      setDecisions(decisionResult.value.records);
      setDecisionTotal(decisionResult.value.total);
      setDecisionPage((current) => Math.min(
        current,
        Math.max(1, Math.ceil(decisionResult.value.total / decisionPageSize)),
      ));
    } else {
      setDecisions([]);
      setDecisionTotal(0);
      setError((current) => current || `Could not load decision ledger: ${decisionResult.reason.message}`);
    }
    const hostResult = outcomes[3];
    if (hostResult.status === 'fulfilled') {
      setHosts(hostResult.value.hosts);
      setHostSnapshots(hostResult.value.snapshots);
      setSelectedHostId((current) =>
        hostResult.value.hosts.some((host) => host.host_id === current)
          ? current
          : hostResult.value.hosts[0]?.host_id ?? '',
      );
    } else {
      setHosts([]);
      setHostSnapshots([]);
      setError((current) => current || `Could not load host utilization: ${hostResult.reason.message}`);
    }
    const placementResult = outcomes[4];
    if (placementResult.status === 'fulfilled') setPlacements(placementResult.value);
    else {
      setPlacements([]);
      setError((current) => current || `Could not load VM placements: ${placementResult.reason.message}`);
    }
  }, [
    activeDatasetId,
    hasMatchingResults,
    selectedPolicy,
    policyRunId,
    migrationPage,
    migrationPageSize,
    eventFilter,
    migrationSearch,
    migrationHostFilter,
    migrationSortBy,
    migrationSortOrder,
    decisionPage,
    decisionPageSize,
    decisionFilter,
    decisionSearch,
    decisionHostFilter,
    decisionSortBy,
    decisionSortOrder,
  ]);

  const refreshRuns = useCallback(async () => {
    const response = await apiRequest<{ runs: RunStatus[] }>('/runs');
    setRuns(response.runs);
  }, []);

  useEffect(() => {
    void loadInitial();
  }, [loadInitial]);

  useEffect(() => {
    let cancelled = false;
    const restoreWorkspace = async () => {
      try {
        const [catalog, runResponse] = await Promise.all([
          apiRequest<NonNullable<typeof datasetInfo>>('/dataset/metadata'),
          apiRequest<{ runs: RunStatus[] }>('/runs'),
        ]);
        if (cancelled) return;
        setDatasetInfo(catalog);
        setActiveDatasetId(catalog.active_dataset_id);
        setRuns(runResponse.runs);
        const latestCompletedRun = runResponse.runs.find(
          (run) =>
            run.status === 'COMPLETED'
            && run.dataset_id === catalog.active_dataset_id,
        );
        if (latestCompletedRun) {
          setSelectedRunId(latestCompletedRun.run_id);
          setCompletedResultRunId(latestCompletedRun.run_id);
          setSelectedPolicy(latestCompletedRun.policy);
          setPage('overview');
        }
      } catch (requestError) {
        if (!cancelled) {
          setError(`Could not restore the existing dataset and run history: ${(requestError as Error).message}`);
        }
      }
    };
    void restoreWorkspace();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const requestId = ++overviewRequestRef.current;
    if (!activeDatasetId || !hasDatasetResults || !policyRunId) {
      setOverview(null);
      setComparison([]);
      return;
    }
    const loadOverview = async () => {
      try {
        const result = await apiRequest<Overview>(
          `/overview${query({ dataset_id: activeDatasetId, run_id: policyRunId })}`,
        );
        if (requestId !== overviewRequestRef.current) return;
        setOverview(result);
        setComparison(result.policy_comparison);
      } catch (requestError) {
        if (requestId !== overviewRequestRef.current) return;
        setOverview(null);
        setComparison([]);
        setError(`Could not load results for this dataset: ${(requestError as Error).message}`);
      }
    };
    void loadOverview();
    return () => {
      if (overviewRequestRef.current === requestId) overviewRequestRef.current += 1;
    };
  }, [
    activeDatasetId,
    hasDatasetResults,
    policyRunId,
  ]);

  useEffect(() => {
    setMigrationPage(1);
    setDecisionPage(1);
    setSelectedVmId('');
    setVmSearch('');
    setPredictionStart('');
    setPredictionEnd('');
    setPredictionSearch('');
    setRunVmId('ALL');
    setSelectedHostId('');
    setHostSearch('');
    setReplayEventKey('');
    setMigrationSearch('');
    setMigrationHostFilter('');
    setEventFilter('');
    setDecisionSearch('');
    setDecisionHostFilter('');
    setDecisionFilter('ACTIONABLE');
  }, [activeDatasetId]);

  useEffect(() => {
    setSavedDatasetId(activeDatasetId ?? '');
  }, [activeDatasetId]);

  useEffect(() => {
    if (activeDatasetId) void loadVms();
  }, [activeDatasetId, loadVms]);

  useEffect(() => {
    void loadPolicyData();
  }, [loadPolicyData]);

  useEffect(() => {
    if (!selectedVmId || !activeDatasetId) {
      setVmDetail(null);
      return;
    }
    let cancelled = false;
    apiRequest<VmDetail>(
      `/vms/${encodeURIComponent(selectedVmId)}${query({
        dataset_id: activeDatasetId,
        run_id: policyRunId || undefined,
        policy: selectedPolicy,
        start_time: predictionStart || undefined,
        end_time: predictionEnd || undefined,
      })}`,
    )
      .then((detail) => {
        if (!cancelled) setVmDetail(detail);
      })
      .catch((requestError: Error) => {
        if (!cancelled) {
          setVmDetail(null);
          setError(`Could not load VM details: ${requestError.message}`);
        }
      });
    return () => { cancelled = true; };
  }, [selectedVmId, activeDatasetId, policyRunId, selectedPolicy, predictionStart, predictionEnd]);

  useEffect(() => {
    setReplayEventKey('');
  }, [activeDatasetId, policyRunId, selectedPolicy]);

  useEffect(() => {
    if (!selectedHostId) {
      setHostSnapshots([]);
      setHostMigrationEvents([]);
      return;
    }
    if (!activeDatasetId || !hasMatchingResults) {
      setHostSnapshots([]);
      setHostMigrationEvents([]);
      return;
    }
    let cancelled = false;
    const resultParams = {
      dataset_id: activeDatasetId,
      ...(policyRunId ? { run_id: policyRunId } : {}),
    };
    setHostSnapshots([]);
    setHostMigrationEvents([]);
    Promise.all([
      apiRequest<{ snapshots: HostRow[] }>(
        `/policies/${selectedPolicy}/hosts${query({ ...resultParams, host_id: selectedHostId })}`,
      ),
      fetchAllPages(
        (page, pageSize) => apiRequest<{ total: number; records: MigrationEvent[] }>(
          `/policies/${selectedPolicy}/migrations${query({
            ...resultParams,
            host_filter: selectedHostId,
            page,
            page_size: pageSize,
          })}`,
        ),
      ),
    ])
      .then(([hostResponse, eventResponse]) => {
        if (cancelled) return;
        setHostSnapshots(hostResponse.snapshots);
        setHostMigrationEvents(eventResponse);
      })
      .catch((requestError: Error) => {
        if (!cancelled) {
          setHostSnapshots([]);
          setHostMigrationEvents([]);
          setError(`Could not load selected host history: ${requestError.message}`);
        }
      });
    return () => { cancelled = true; };
  }, [selectedHostId, selectedPolicy, policyRunId, activeDatasetId, hasMatchingResults]);

  useEffect(() => {
    if (!selectedRunId) return;
    let cancelled = false;
    let interval = 0;
    const poll = async () => {
      try {
        const [status, logs] = await Promise.all([
          apiRequest<{ run: RunStatus }>(`/runs/${selectedRunId}`),
          apiRequest<{ stdout: string; stderr: string }>(`/runs/${selectedRunId}/logs`),
        ]);
        if (cancelled) return;
        setRuns((current) => current.map((run) => run.run_id === selectedRunId ? status.run : run));
        setRunLogs(logs);
        if (status.run.status === 'COMPLETED') {
          if (status.run.dataset_id === activeDatasetId) {
            setSelectedPolicy(status.run.policy);
              setCompletedResultRunId(status.run.run_id);
              setPage('overview');
            }
          setSuccess(
            status.run.dataset_id === activeDatasetId
              ? `Run ${status.run.run_id} completed. Its isolated output is available for this dataset.`
              : `Run ${status.run.run_id} completed for dataset ${status.run.dataset_id}. Select that dataset to view its isolated output.`,
          );
          await refreshRuns();
        } else if (['FAILED', 'TIMEOUT', 'INTERRUPTED'].includes(status.run.status)) {
          setError(status.run.error || `Simulation run ended with status ${status.run.status}.`);
          await refreshRuns();
        } else {
          return;
        }
        if (interval) window.clearInterval(interval);
      } catch (requestError) {
        if (!cancelled) setError(`Run status polling failed: ${(requestError as Error).message}`);
      }
    };
    void poll();
    interval = window.setInterval(() => void poll(), 2500);
    return () => {
      cancelled = true;
      if (interval) window.clearInterval(interval);
    };
  }, [selectedRunId, refreshRuns, activeDatasetId]);

  const sortedComparison = useMemo(() => {
    const search = benchmarkSearch.trim().toLowerCase();
    return [...comparison]
      .filter((row) => !search || row.policy.toLowerCase().includes(search))
      .sort((left, right) => {
        const a = numeric(left[sortMetric]);
        const b = numeric(right[sortMetric]);
        if (a === null && b === null) return left.policy.localeCompare(right.policy);
        if (a === null) return 1;
        if (b === null) return -1;
        return (a - b) * (sortDirection === 'asc' ? 1 : -1);
      });
  }, [comparison, benchmarkSearch, sortMetric, sortDirection]);

  const hostVms = useMemo(
    () => placements.filter((vm) => String(vm.final_host ?? '') === selectedHostId),
    [placements, selectedHostId],
  );
  const selectedHost = hosts.find((host) => host.host_id === selectedHostId) ?? null;
  const hostMigrations = hostMigrationEvents.filter(
    (event) => event.source_host === selectedHostId || event.target_host === selectedHostId,
  );
  const replayEvent = hostMigrations.find((event) =>
    event.event === 'SUCCEEDED' &&
    `${event.simulation_time}:${event.source_vm_id}:${event.source_host}:${event.target_host}` === replayEventKey,
  );
  const latestPrediction = vmDetail?.prediction_series[vmDetail.prediction_series.length - 1];
  const comparisonForPolicy = comparison.find((row) => row.policy === selectedPolicy);
  const topologyNodes = useMemo(() => {
    if (!selectedHost) return [];
    if (replayEvent) {
      return [
        {
          id: `host:${replayEvent.source_host}`,
          position: { x: 20, y: 140 },
          data: { label: `Source host ${replayEvent.source_host}` },
          style: { background: '#0f2942', color: '#e0f2fe', border: '1px solid #38bdf8', width: 190 },
        },
        {
          id: `vm:${replayEvent.source_vm_id}`,
          position: { x: 330, y: 140 },
          data: { label: `VM ${replayEvent.source_vm_id} · SUCCEEDED` },
          style: { background: '#174234', color: '#dcfce7', border: '1px solid #34d399', width: 190 },
        },
        {
          id: `host:${replayEvent.target_host}`,
          position: { x: 650, y: 140 },
          data: { label: `Destination host ${replayEvent.target_host}` },
          style: { background: '#0f2942', color: '#e0f2fe', border: '1px solid #38bdf8', width: 210 },
        },
      ];
    }
    const hostNode = {
      id: `host:${selectedHost.host_id}`,
      position: { x: 30, y: 140 },
      data: {
        label: `Host ${selectedHost.host_id} · CPU ${display(numeric(selectedHost.current_cpu_fraction) === null ? null : numeric(selectedHost.current_cpu_fraction)! * 100, 1)}%`,
      },
      style: { background: '#0f2942', color: '#e0f2fe', border: '1px solid #38bdf8', width: 220 },
    };
    const vmNodes = hostVms.slice(0, 40).map((vm, index) => ({
      id: `vm:${vm.source_vm_id}`,
      position: { x: 330 + (index % 4) * 185, y: 40 + Math.floor(index / 4) * 86 },
      data: { label: `VM ${vm.source_vm_id}` },
      style: { background: '#18263b', color: '#cbd5e1', border: '1px solid #475569', width: 130 },
    }));
    return [hostNode, ...vmNodes];
  }, [selectedHost, hostVms, replayEvent]);

  const topologyEdges = useMemo(
    () => replayEvent
      ? [
          {
            id: `move-out:${replayEvent.source_vm_id}:${replayEvent.simulation_time}`,
            source: `host:${replayEvent.source_host}`,
            target: `vm:${replayEvent.source_vm_id}`,
            animated: false,
            style: { stroke: '#38bdf8' },
          },
          {
            id: `move-in:${replayEvent.source_vm_id}:${replayEvent.simulation_time}`,
            source: `vm:${replayEvent.source_vm_id}`,
            target: `host:${replayEvent.target_host}`,
            animated: true,
            style: { stroke: '#34d399' },
          },
        ]
      : hostVms.slice(0, 40).map((vm) => ({
          id: `placement:${selectedHostId}:${vm.source_vm_id}`,
          source: `host:${selectedHostId}`,
          target: `vm:${vm.source_vm_id}`,
          animated: false,
          style: { stroke: '#38bdf8' },
        })),
    [hostVms, selectedHostId, replayEvent],
  );

  const chooseUploadFile = (file: File | null) => {
    setUploadFile(file);
    setUploadValidation(null);
    setUploadResult(null);
    setActiveDatasetId(null);
    setCompletedResultRunId('');
    setSelectedRunId((current) => runs.find((run) => run.run_id === current && ['PENDING', 'RUNNING'].includes(run.status))?.run_id ?? '');
    setOverview(null);
    setPolicySummary({});
    setComparison([]);
    setMigrationEvents([]);
    setDecisions([]);
    setHosts([]);
    setPlacements([]);
    setError('');
    setSuccess('');
    if (file && !file.name.toLowerCase().endsWith('.csv')) {
      setError('Choose a file with a .csv extension.');
      return;
    }
    if (file && file.size > (datasetInfo?.upload_max_bytes ?? 200 * 1024 * 1024)) {
      setError('The selected CSV exceeds the 200 MiB upload limit.');
    }
  };

  const selectRun = (runId: string) => {
    setSelectedRunId(runId);
    const run = runs.find((item) => item.run_id === runId);
    if (run?.status === 'COMPLETED' && run.dataset_id === activeDatasetId) {
      setSelectedPolicy(run.policy);
      setCompletedResultRunId(run.run_id);
    }
  };

  const switchSavedDataset = async () => {
    const dataset = datasetInfo?.datasets.find((item) => item.dataset_id === savedDatasetId);
    if (!dataset || dataset.dataset_id === activeDatasetId) return;
    setBusy(true);
    setError('');
    setSuccess('');
    try {
      await apiRequest('/dataset/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: dataset.dataset_id }),
      });
      const [catalog, runResponse] = await Promise.all([
        apiRequest<NonNullable<typeof datasetInfo>>('/dataset/metadata'),
        apiRequest<{ runs: RunStatus[] }>('/runs'),
      ]);
      const latestRun = runResponse.runs.find(
        (run) => run.status === 'COMPLETED' && run.dataset_id === dataset.dataset_id,
      );
      setDatasetInfo(catalog);
      setRuns(runResponse.runs);
      setActiveDatasetId(dataset.dataset_id);
      setSelectedRunId('');
      setCompletedResultRunId(latestRun?.run_id ?? '');
      setSelectedPolicy(latestRun?.policy ?? 'ML_PROACTIVE');
      setUploadFile(null);
      setUploadResult(null);
      setUploadValidation(null);
      setOverview(null);
      setComparison([]);
      setPolicySummary({});
      setMigrationEvents([]);
      setMigrationTotal(0);
      setDecisions([]);
      setDecisionTotal(0);
      setHosts([]);
      setHostSnapshots([]);
      setHostMigrationEvents([]);
      setPlacements([]);
      setRunLogs({ stdout: '', stderr: '' });
      setPage('prediction');
      setSuccess(`Selected ${dataset.filename}.`);
    } catch (requestError) {
      setSavedDatasetId(activeDatasetId ?? '');
      setError(`Could not switch datasets: ${(requestError as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const validateDataset = async () => {
    if (!uploadFile) return;
    setUploadValidation(null);
    setUploadResult(null);
    if (uploadFile.size > (datasetInfo?.upload_max_bytes ?? 200 * 1024 * 1024)) {
      setError('The selected CSV exceeds the 200 MiB upload limit.');
      return;
    }
    setBusy(true);
    setError('');
    setSuccess('');
    try {
      const form = new FormData();
      form.append('file', uploadFile);
      const response = await apiRequest<{
        dataset: DatasetInfo;
        duplicate?: boolean;
      }>('/dataset/upload', {
        method: 'POST',
        body: form,
      });
      const dataset = response.dataset;
      const datasetKind = dataset.dataset_kind ?? 'raw_or_incompatible_csv';
      const formatNote = datasetKind === 'prediction_csv'
        ? 'Prepared forecasts and risk values are recorded inputs; no fresh inference is performed.'
        : datasetKind === 'raw_workload_csv'
          ? 'Raw Bitbrains workload schema matches the training preprocessing; model inference is required.'
          : datasetKind === 'incompatible_raw_workload_csv'
            ? 'Alibaba raw data is preserved, but its memory and disk-space percentages are incompatible with the trained model inputs.'
          : 'The CSV schema is not compatible with the raw workload or prepared prediction pipeline.';
      setUploadValidation({
        filename: dataset.filename,
        size_bytes: dataset.size_bytes,
        validation: {
          ...dataset.validation,
          row_count: dataset.row_count,
          vm_count: dataset.vm_count,
          columns: dataset.columns,
        },
        content_sha256: '',
        duplicate_dataset_id: response.duplicate ? dataset.dataset_id : null,
        preview: [],
        dataset_kind: datasetKind,
        format_note: formatNote,
        inference_available: Boolean(datasetInfo?.model_inference_available),
        dataset_id: dataset.dataset_id,
      });
      setUploadResult(dataset);
      setDatasetInfo((current) => current ? {
        ...current,
        datasets: [
          dataset,
          ...current.datasets.filter((item) => item.dataset_id !== dataset.dataset_id),
        ],
      } : current);
      setSuccess(
        `${dataset.filename} uploaded and validated: ${dataset.row_count.toLocaleString()} rows, `
        + `${dataset.vm_count.toLocaleString()} VMs.`,
      );
    } catch (requestError) {
      setError(`Upload or validation failed: ${(requestError as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const validateLocalDataset = async () => {
    setUploadValidation(null);
    setUploadResult(null);
    setBusy(true);
    setError('');
    setSuccess('');
    try {
      const response = await apiRequest<{
        dataset: DatasetInfo;
        validation_duration_seconds: number;
        format_note: string;
      }>('/dataset/local/validate', { method: 'POST' });
      const dataset = response.dataset;
      setUploadValidation({
        filename: dataset.filename,
        size_bytes: dataset.size_bytes,
        validation: {
          ...dataset.validation,
          row_count: dataset.row_count,
          vm_count: dataset.vm_count,
          columns: dataset.columns,
        },
        content_sha256: '',
        duplicate_dataset_id: null,
        preview: dataset.preview,
        dataset_kind: 'raw_workload_csv',
        format_note: response.format_note,
        inference_available: Boolean(datasetInfo?.model_inference_available),
        dataset_id: dataset.dataset_id,
      });
      setUploadResult(dataset);
      setDatasetInfo((current) => current ? {
        ...current,
        datasets: [
          dataset,
          ...current.datasets.filter((item) => item.dataset_id !== dataset.dataset_id),
        ],
      } : current);
      setSuccess(
        `${dataset.filename} validated in place: ${dataset.row_count.toLocaleString()} rows, `
        + `${dataset.vm_count.toLocaleString()} VMs. The file was not uploaded or copied.`,
      );
    } catch (requestError) {
      setError(`Local CSV validation failed: ${(requestError as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const submitUploadedDataset = async () => {
    const dataset = uploadResult;
    if (!dataset || !uploadValidation?.validation.valid) {
      return;
    }
    if (
      dataset.dataset_kind !== 'prediction_csv'
      && dataset.dataset_kind !== 'raw_workload_csv'
      && dataset.dataset_kind !== 'incompatible_raw_workload_csv'
    ) {
      setError(uploadValidation.format_note);
      return;
    }
    if (dataset.dataset_kind === 'incompatible_raw_workload_csv') {
      setBusy(true);
      setError('');
      setSuccess('');
      try {
        await apiRequest('/dataset/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_id: dataset.dataset_id }),
        });
        const catalog = await apiRequest<NonNullable<typeof datasetInfo>>('/dataset/metadata');
        setDatasetInfo(catalog);
        setActiveDatasetId(dataset.dataset_id);
        setCompletedResultRunId('');
        setOverview(null);
        setComparison([]);
        setPolicySummary({});
        setMigrationEvents([]);
        setDecisions([]);
        setHosts([]);
        setPlacements([]);
        setSuccess(
          `${dataset.filename} was kept as raw workload data. Inference and CloudSim are disabled because required model channels are absent.`,
        );
        setPage('dataset');
      } catch (requestError) {
        setError(`Could not select the preserved raw dataset: ${(requestError as Error).message}`);
      } finally {
        setBusy(false);
      }
      return;
    }
    setBusy(true);
    setError('');
    setSuccess('');
    try {
      await apiRequest('/dataset/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: dataset.dataset_id }),
      });
      const catalog = await apiRequest<NonNullable<typeof datasetInfo>>('/dataset/metadata');
      setDatasetInfo(catalog);
      setActiveDatasetId(dataset.dataset_id);
      setCompletedResultRunId('');
      setOverview(null);
      setComparison([]);
      setPolicySummary({});
      setMigrationEvents([]);
      setDecisions([]);
      setHosts([]);
      setPlacements([]);
      setRunLogs({ stdout: '', stderr: '' });
      const runResponse = await apiRequest<{ run: RunStatus }>('/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy: runPolicy, dataset_id: dataset.dataset_id }),
      });
      setSelectedRunId(runResponse.run.run_id);
      setRuns((current) => [
        runResponse.run,
        ...current.filter((run) => run.run_id !== runResponse.run.run_id),
      ]);
      setSuccess(`Pipeline submitted as ${runResponse.run.run_id}. Stage status and logs are available below.`);
      setPage('simulation');
    } catch (requestError) {
      setError(`Pipeline could not start: ${(requestError as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const storeDataset = () => {
    const dataset = uploadResult;
    if (!dataset || !uploadValidation?.validation.valid) return;
    if (
      dataset.dataset_kind !== 'prediction_csv'
      && dataset.dataset_kind !== 'raw_workload_csv'
      && dataset.dataset_kind !== 'incompatible_raw_workload_csv'
    ) {
      setError(uploadValidation.format_note);
      return;
    }
    if (dataset.dataset_kind === 'incompatible_raw_workload_csv') {
      void submitUploadedDataset();
      return;
    }
    setConfirmation({
      title: 'Run pipeline?',
      message: `Run ${runPolicy} on ${dataset.filename} (${dataset.row_count.toLocaleString()} rows)?`,
      confirmLabel: 'Run pipeline',
      onConfirm: () => { void submitUploadedDataset(); },
    });
  };

  const submitRun = async () => {
    if (!activeDatasetId) return;
    setBusy(true);
    setError('');
    setSuccess('');
    setRunLogs({ stdout: '', stderr: '' });
    setCompletedResultRunId('');
    setOverview(null);
    setComparison([]);
    try {
      const response = await apiRequest<{ run: RunStatus }>('/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          policy: runPolicy,
          dataset_id: activeDatasetId,
          vm_ids: parseRunVmSelection(runVmId).vmIds,
          start_time: runStartTime || undefined,
          end_time: runEndTime || undefined,
          host_count: Number(runHostCount),
          host_ram_mb: Number(runHostRamMb),
          host_pes: Number(runHostPes),
          cpu_limit: Number(runCpuLimit),
          ram_limit: Number(runRamLimit),
          risk_threshold: Number(runRiskThreshold),
        }),
      });
      setSelectedRunId(response.run.run_id);
      setCompletedResultRunId('');
      setRuns((current) => [response.run, ...current.filter((run) => run.run_id !== response.run.run_id)]);
      setSuccess(`Run submitted: ${response.run.run_id}. Status and run logs are polling.`);
    } catch (requestError) {
      setError(`Could not start the simulation: ${(requestError as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const startRun = () => {
    if (!activeDatasetId || !datasetInfo?.datasets.find((item) => item.dataset_id === activeDatasetId)) {
      setError('Upload and validate a supported raw workload or prepared prediction CSV before starting the pipeline.');
      return;
    }
    const dataset = datasetInfo.datasets.find((item) => item.dataset_id === activeDatasetId);
    if (!dataset) return;
    if (dataset.dataset_kind === 'incompatible_raw_workload_csv') {
      setError(uploadValidation?.format_note ?? 'This raw dataset cannot be simulated.');
      return;
    }
    const selectedRunVmId = runVmId.trim();
    const vmSelection = parseRunVmSelection(selectedRunVmId);
    if (vmSelection.error) {
      setError(vmSelection.error);
      return;
    }
    const availableVmIds = new Set(dataset.vm_ids ?? []);
    const unknownVmIds = (vmSelection.vmIds ?? []).filter((vmId) =>
      availableVmIds.size
        ? !availableVmIds.has(vmId)
        : !vms.some((vm) => vm.source_vm_id === vmId),
    );
    if (unknownVmIds.length) {
      setError(`VM IDs not present in the selected dataset: ${unknownVmIds.join(', ')}.`);
      return;
    }
    const vmSelectionLabel = vmSelection.vmIds?.length
      ? `VMs ${vmSelection.vmIds.join(', ')}`
      : 'all VMs';
    setConfirmation({
      title: 'Confirm CloudSim run',
      message: `Run all five policies on ${dataset.filename} using ${vmSelectionLabel} and the selected time window? Threshold ${runRiskThreshold}; hosts ${runHostCount} × ${runHostRamMb} MB RAM × ${runHostPes} PEs.`,
      confirmLabel: 'Start CloudSim',
      onConfirm: () => { void submitRun(); },
    });
  };

  const confirmPendingAction = () => {
    if (!confirmation) return;
    const action = confirmation.onConfirm;
    setConfirmation(null);
    action();
  };

  const resetRunView = () => {
    if (selectedRun && ['PENDING', 'RUNNING'].includes(selectedRun.status)) {
      setRunLogs({ stdout: '', stderr: '' });
      setSuccess('The log view was cleared; the active simulator process continues to be monitored.');
      return;
    }
    setSelectedRunId('');
    setCompletedResultRunId('');
    setOverview(null);
    setComparison([]);
    setRunLogs({ stdout: '', stderr: '' });
    setSuccess('Run view reset. Completed run records and output files were preserved.');
  };

  const refreshAll = async () => {
    setError('');
    try {
      await loadInitial();
      if (selectedRunId) await refreshRuns();
      setSuccess('Backend connection and selected run status refreshed.');
    } catch (requestError) {
      setError(`Refresh failed: ${(requestError as Error).message}`);
    }
  };

  const onTopologyNodeClick = (_event: React.MouseEvent, node: { id: string }) => {
    if (node.id.startsWith('host:')) setSelectedHostId(node.id.slice('host:'.length));
    if (node.id.startsWith('vm:')) {
      setSelectedVmId(node.id.slice('vm:'.length));
      setPage('results-prediction');
    }
  };

  const activeSection: SectionKey =
    page === 'dataset' || page === 'prediction' || page === 'simulation'
      ? 'pipeline'
      : 'results';
  const sectionTitle = NAV.find((item) => item.key === activeSection)?.label ?? 'Pipeline';
  const workflowSteps = activeSection === 'pipeline'
    ? [
        { key: 'dataset' as const, label: 'Upload & validate' },
        { key: 'prediction' as const, label: 'Forecasts' },
        { key: 'simulation' as const, label: 'CloudSim' },
      ]
    : activeSection === 'results' && hasMatchingResults
      ? [
          { key: 'overview' as const, label: 'Run summary' },
          { key: 'results-prediction' as const, label: 'VM forecasts' },
          { key: 'explorer' as const, label: 'Hosts & placement' },
          { key: 'migrations' as const, label: 'Migrations & ledger' },
          { key: 'benchmark' as const, label: 'Compare runs' },
        ]
      : [];

  const metricChartRows = sortedComparison.map((row) => ({
    policy: row.policy,
    value: numeric(row[comparisonMetric]),
  }));

  const overloadSnapshots = numeric(
    policySummary.host_overload_snapshots ?? comparisonForPolicy?.host_overload_snapshots,
  );
  const energyEstimate = numeric(
    policySummary.energy_kwh ?? comparisonForPolicy?.energy_kwh,
  );
  const summaryCards = [
    { label: 'Virtual machines', value: display(hasMatchingResults ? policySummary.unique_vms : activeDataset?.vm_count, 0), detail: hasMatchingResults ? 'Selected run' : 'Selected dataset', tone: 'blue' as const },
    { label: 'Configured hosts', value: display(hasMatchingResults ? policySummary.configured_hosts : null, 0), detail: 'Configured simulation assumption', tone: 'blue' as const },
    { label: 'Migration requests', value: display(hasMatchingResults ? policySummary.migration_requests ?? comparisonForPolicy?.migration_requests : null, 0), detail: `Policy ${selectedPolicy}`, tone: 'amber' as const },
    { label: 'Successful migrations', value: display(hasMatchingResults ? policySummary.successful_migrations ?? comparisonForPolicy?.successful_migrations : null, 0), detail: 'Confirmed by matching simulator output', tone: 'green' as const },
    ...(overloadSnapshots === null ? [] : [{
      label: 'Host overload snapshots',
      value: display(overloadSnapshots, 0),
      detail: 'Unique host/time CPU-or-RAM threshold exceedances; simulation proxy',
      tone: 'red' as const,
    }]),
    ...(energyEstimate === null ? [] : [{
      label: 'Energy estimate',
      value: `${display(energyEstimate, 2)} kWh`,
      detail: 'Approximate simulated energy',
      tone: 'blue' as const,
    }]),
  ];

  const renderPage = () => {
    if (activeSection === 'results' && !hasMatchingResults) {
      return <section className="panel section-panel results-empty"><EmptyState title="No results yet. Upload a dataset and run the pipeline." detail="Results appear only after a successful CloudSim run for the currently selected dataset. Saved v10 outputs are not shown here." /></section>;
    }
    if (page === 'overview') {
      return (
        <>
          <section className="metrics-grid">
            {summaryCards.map((card) => <MetricCard key={card.label} {...card} />)}
          </section>
          <div className="notice">
            <strong>SLA status: NOT_MEASURED</strong>
            <span>Host overload snapshots are simulation proxies, not observed SLA breaches.</span>
          </div>
          <section className="panel section-panel">
            <div className="section-header">
              <div>
                <div className="eyebrow">Source: {resultSourceLabel}</div>
                <h2>Overview</h2>
              </div>
              <div className="control-group">
                <label htmlFor="overview-policy">Run policy</label>
                <select id="overview-policy" value={selectedPolicy} disabled>
                  <option>{selectedPolicy}</option>
                </select>
              </div>
            </div>
            {!hasMatchingResults && <EmptyState title="No matching simulation results yet" detail="Prediction dataset metadata is available, but simulation metrics appear only after a run completes for this dataset." />}
            <div className="content-grid two-col">
              {comparison.some((row) => numeric(row.energy_kwh) !== null) && <article className="chart-panel">
                <div className="chart-title-row">
                  <h3>Approximate energy by policy</h3>
                  <span className="badge info">Simulation estimate · kWh</span>
                </div>
                <div className="chart-area">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparison.filter((row) => numeric(row.energy_kwh) !== null).map((row) => ({ policy: row.policy, value: numeric(row.energy_kwh) }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#26354a" /><XAxis dataKey="policy" stroke="#94a3b8" tick={{ fontSize: 10 }} /><YAxis stroke="#94a3b8" /><Tooltip formatter={(value) => [`${display(value, 3)} kWh`, 'Energy estimate']} /><Bar dataKey="value" name="Energy estimate (kWh)" fill="#38bdf8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>}
              <article className="chart-panel">
                <h3>Migration counts by policy</h3>
                <div className="chart-area">
                  {comparison.length ? <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparison.map((row) => ({ policy: row.policy, requests: numeric(row.migration_requests), succeeded: numeric(row.successful_migrations) }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#26354a" /><XAxis dataKey="policy" stroke="#94a3b8" tick={{ fontSize: 10 }} /><YAxis stroke="#94a3b8" allowDecimals={false} /><Tooltip formatter={(value, name) => [display(value, 0), name]} /><Legend /><Bar dataKey="requests" name="Migration requests" fill="#f59e0b" /><Bar dataKey="succeeded" name="Successful migrations" fill="#34d399" />
                    </BarChart>
                  </ResponsiveContainer> : <EmptyState title="Migration counts unavailable" detail="No comparable policy rows are available for this result source." />}
                </div>
              </article>
            </div>
          </section>
          <section className="content-grid two-col">
            <article className="panel">
              <div className="section-header compact"><h3>Recent migration events</h3><button className="text-button" onClick={() => setPage('migrations')}>Open explorer</button></div>
              <DataTable
                rows={migrationEvents.slice(0, 8)}
                columns={[
                  { key: 'source_vm_id', label: 'VM' },
                  { key: 'event', label: 'Event', render: (row) => <span className={eventClass(String(row.event))}>{String(row.event)}</span> },
                  { key: 'source_host', label: 'Source host' },
                  { key: 'target_host', label: 'Destination' },
                ]}
              />
            </article>
            <article className="panel">
            <div className="section-header compact"><h3>Recent simulation decisions</h3><button className="text-button" onClick={() => setPage('migrations')}>Open ledger</button></div>
              <DataTable
                rows={decisions.slice(0, 8)}
                columns={[
                  { key: 'timestamp', label: 'Time' },
                  { key: 'vm_id', label: 'VM' },
                  { key: 'decision', label: 'Decision' },
                  { key: 'risk_probability', label: 'Risk', render: (row) => display(row.risk_probability) },
                ]}
              />
            </article>
          </section>
        </>
      );
    }

    if (page === 'prediction' || page === 'results-prediction') {
      const series = vmDetail?.prediction_series ?? [];
      const filteredSeries = series.filter((row) => {
        const time = String(row.decision_time ?? '').replace(' ', 'T').slice(0, 16);
        const matchesRange = (!predictionStart || time >= predictionStart)
          && (!predictionEnd || time <= predictionEnd);
        const term = predictionSearch.trim().toLocaleLowerCase();
        const matchesSearch = !term || Object.values(row).some((value) =>
          String(value ?? '').toLocaleLowerCase().includes(term),
        );
        return matchesRange && matchesSearch;
      });
      const latestInRange = filteredSeries[filteredSeries.length - 1];
      const cpu = filteredSeries.map((row) => ({
        time: String(row.decision_time ?? ''),
        current: numeric(row.current_cpu_pct),
        forecast: numeric(row.forecast_cpu_max),
        simulatorClampedForecast: numeric(row.simulator_forecast_cpu_pct),
      }));
      const memory = filteredSeries.map((row) => ({
        time: String(row.decision_time ?? ''),
        current: numeric(row.current_memory_kb) === null ? null : Number(row.current_memory_kb) / 1024,
        forecast: numeric(row.forecast_memory_max) === null ? null : Number(row.forecast_memory_max) / 1024,
      }));
      const risk = numeric(latestInRange?.risk_probability);
      const threshold = numeric(latestInRange?.risk_threshold);
      const crosses = risk !== null && threshold !== null ? risk >= threshold : null;
      const isFreshInference = vmDetail?.prediction_data_mode === 'fresh_model_inference';
      const isRawDataset = activeDataset?.dataset_kind === 'raw_workload_csv';
      const inferenceStatus = isFreshInference
        ? 'Fresh inference'
        : activeDataset?.dataset_kind === 'prediction_csv'
          ? 'Recorded predictions'
          : activeDataset?.dataset_kind === 'incompatible_raw_workload_csv'
            ? 'Incompatible raw inputs'
            : 'Inference pending';
      const inferenceDisabledReason = !datasetInfo?.model_inference_available
        ? inferenceNote || 'Required model artifacts have not loaded successfully.'
        : !isRawDataset
          ? 'Fresh inference requires a compatible raw workload dataset.'
          : undefined;
      const alertText = crosses === null ? 'Unavailable' : crosses ? 'ALERT' : 'NO ALERT';
      return (
        <>
          <section className="panel section-panel">
            <div className="section-header">
              <div><h2>Forecasts & VM</h2></div>
              <div className="button-row">
                <span className={`badge ${isFreshInference ? 'success' : 'info'}`}>{inferenceStatus}</span>
                <button
                  className="button secondary"
                    disabled={!isRawDataset || !datasetInfo?.model_inference_available || inferenceBusy || !selectedVmId}
                  title={inferenceDisabledReason}
                  onClick={() => void runModelInference()}
                >
                    {inferenceBusy ? 'Running inference…' : 'Infer selected VM'}
                </button>
              </div>
            </div>
            <div className="toolbar">
              <label className="control-group grow"><span>Search VM</span><input value={vmSearch} onChange={(event) => setVmSearch(event.target.value)} placeholder="Search VM ID" /></label>
              <label className="control-group"><span>Select VM</span><select value={selectedVmId} onChange={(event) => setSelectedVmId(event.target.value)}>{vms.map((vm) => <option key={vm.source_vm_id} value={vm.source_vm_id}>{vm.source_vm_id}</option>)}</select></label>
              <label className="control-group grow"><span>Search forecast rows</span><input value={predictionSearch} onChange={(event) => setPredictionSearch(event.target.value)} placeholder="Search timestamp or value" /></label>
              <label className="control-group"><span>From (local)</span><input type="datetime-local" value={predictionStart} onChange={(event) => setPredictionStart(event.target.value)} /></label>
              <label className="control-group"><span>To (local)</span><input type="datetime-local" value={predictionEnd} onChange={(event) => setPredictionEnd(event.target.value)} /></label>
              <button className="button secondary small" onClick={() => { setPredictionStart(''); setPredictionEnd(''); setPredictionSearch(''); }}>Clear filters</button>
              <span className="muted">{vmTotal.toLocaleString()} matching VMs</span>
            </div>
            <p className="muted small-copy">{filteredSeries.length.toLocaleString()} of {series.length.toLocaleString()} records</p>
          </section>
          {vmDetail && latestInRange ? (
            <>
              <section className="metrics-grid four">
                <MetricCard label="Current CPU" value={`${display(latestInRange.current_cpu_pct)}%`} detail={`Forecast max ${display(latestInRange.forecast_cpu_max)}%`} />
                <MetricCard label="Latest memory" value={`${display(numeric(latestInRange.current_memory_kb) === null ? null : Number(latestInRange.current_memory_kb) / 1024)} MB`} detail={`Forecast max ${display(numeric(latestInRange.forecast_memory_max) === null ? null : Number(latestInRange.forecast_memory_max) / 1024)} MB`} tone="green" />
                <MetricCard label={`${isFreshInference ? 'Fresh' : 'Recorded'} risk probability`} value={display(risk, 4)} detail={`Threshold ${display(threshold, 2)}`} tone={crosses ? 'red' : 'blue'} />
                <MetricCard label={isFreshInference ? 'Model alert' : 'Recorded alert'} value={alertText} tone={crosses ? 'amber' : 'green'} />
              </section>
              <section className="content-grid two-col">
                <article className="panel chart-panel">
                  <h3>Current and forecast CPU (%)</h3>
                  <div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={cpu}><CartesianGrid strokeDasharray="3 3" stroke="#26354a" /><XAxis dataKey="time" hide /><YAxis stroke="#94a3b8" /><Tooltip /><Legend /><Line dataKey="current" name="Observed CPU (raw)" stroke="#38bdf8" dot={false} /><Line dataKey="forecast" name="Forecast CPU (raw)" stroke="#f59e0b" dot={false} /><Line dataKey="simulatorClampedForecast" name="Forecast after simulator clamp" stroke="#a78bfa" dot={false} /></LineChart></ResponsiveContainer></div>
                </article>
                <article className="panel chart-panel">
                  <h3>Current and forecast memory (MB)</h3>
                  <div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={memory}><CartesianGrid strokeDasharray="3 3" stroke="#26354a" /><XAxis dataKey="time" hide /><YAxis stroke="#94a3b8" /><Tooltip /><Legend /><Line dataKey="current" name="Current memory" stroke="#34d399" dot={false} /><Line dataKey="forecast" name="Forecast maximum" stroke="#a78bfa" dot={false} /></LineChart></ResponsiveContainer></div>
                </article>
              </section>
              {vmDetail.disk_forecasts_available && (
                <section className="panel section-panel">
                  <h3>Disk I/O (forecast fields when present)</h3>
                  <DataTable rows={filteredSeries} resetKey={`${selectedVmId}:${predictionStart}:${predictionEnd}:${predictionSearch}`} columns={[
                    { key: 'decision_time', label: 'Time' },
                    { key: 'current_disk_read_kbps', label: 'Read kB/s' },
                    { key: 'forecast_disk_read_max', label: 'Forecast read max' },
                    { key: 'current_disk_write_kbps', label: 'Write kB/s' },
                    { key: 'forecast_disk_write_max', label: 'Forecast write max' },
                  ]} />
                </section>
              )}
            </>
          ) : <EmptyState title={loading ? 'Loading VM data' : 'No VM selected'} detail={isRawDataset ? 'Run model inference to generate forecasts for this raw workload.' : 'Select a VM with recorded prediction rows in the active dataset.'} />}
        </>
      );
    }

    if (page === 'explorer') {
      const visibleHosts = hosts.filter((host) => host.host_id.toLowerCase().includes(hostSearch.toLowerCase()));
      const snapshotChart = hostSnapshots.map((snapshot) => ({
        timestamp: snapshot.timestamp,
        cpuCurrent: numeric(snapshot.current_cpu_fraction) === null ? null : Number(snapshot.current_cpu_fraction) * 100,
        cpuForecast: numeric(snapshot.forecast_cpu_fraction) === null ? null : Number(snapshot.forecast_cpu_fraction) * 100,
        ramCurrent: numeric(snapshot.current_ram_fraction) === null ? null : Number(snapshot.current_ram_fraction) * 100,
        ramForecast: numeric(snapshot.forecast_ram_fraction) === null ? null : Number(snapshot.forecast_ram_fraction) * 100,
      }));
      return (
        <>
          <section className="panel section-panel">
            <div className="section-header"><div><div className="eyebrow">Source: {resultSourceLabel}</div><h2>Host & VM Explorer</h2></div><span className="badge info">{hosts.length} hosts</span></div>
            {!hasMatchingResults && <EmptyState title="No matching simulation results yet" detail="Host placement and utilization data are not inferred from prediction rows. Run a simulation for this dataset first." />}
            <div className="explorer-layout">
              <aside className="host-list">
                <input value={hostSearch} onChange={(event) => setHostSearch(event.target.value)} placeholder="Search host ID" aria-label="Search hosts" />
                <div className="host-scroll">
                  {visibleHosts.map((host) => (
                    <button key={host.host_id} className={`host-option ${host.host_id === selectedHostId ? 'selected' : ''}`} onClick={() => setSelectedHostId(host.host_id)}>
                      <span>Host {host.host_id}</span>
                      <small>CPU {display(numeric(host.current_cpu_fraction) === null ? null : Number(host.current_cpu_fraction) * 100, 1)}% · RAM {display(numeric(host.current_ram_fraction) === null ? null : Number(host.current_ram_fraction) * 100, 1)}%</small>
                    </button>
                  ))}
                </div>
              </aside>
              <div className="explorer-main">
                {selectedHost ? (
                  <>
                    <div className="metrics-grid four compact-grid">
                      <MetricCard label="Current CPU" value={`${display(numeric(selectedHost.current_cpu_fraction) === null ? null : Number(selectedHost.current_cpu_fraction) * 100, 1)}%`} detail={`Forecast ${display(numeric(selectedHost.forecast_cpu_fraction) === null ? null : Number(selectedHost.forecast_cpu_fraction) * 100, 1)}%`} />
                      <MetricCard label="Current RAM" value={`${display(numeric(selectedHost.current_ram_fraction) === null ? null : Number(selectedHost.current_ram_fraction) * 100, 1)}%`} detail={`Forecast ${display(numeric(selectedHost.forecast_ram_fraction) === null ? null : Number(selectedHost.forecast_ram_fraction) * 100, 1)}%`} tone="green" />
                      <MetricCard label="RAM used" value={`${display(selectedHost.current_ram_mb)} MB`} detail={`Forecast ${display(selectedHost.forecast_ram_mb)} MB`} />
                      <MetricCard label="VM count at snapshot" value={display(selectedHost.vm_count_at_snapshot, 0)} />
                    </div>
                    <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={snapshotChart}><CartesianGrid strokeDasharray="3 3" stroke="#26354a" /><XAxis dataKey="timestamp" hide /><YAxis stroke="#94a3b8" unit="%" /><Tooltip /><Legend /><Line dataKey="cpuCurrent" name="Current CPU" stroke="#38bdf8" dot={false} /><Line dataKey="cpuForecast" name="Forecast CPU" stroke="#f59e0b" dot={false} /><Line dataKey="ramCurrent" name="Current RAM" stroke="#34d399" dot={false} /><Line dataKey="ramForecast" name="Forecast RAM" stroke="#a78bfa" dot={false} /></LineChart></ResponsiveContainer></div>
                    <div className="topology-panel">
                      <ReactFlowProvider><ReactFlow nodes={topologyNodes} edges={topologyEdges} fitView nodesDraggable={false} onNodeClick={onTopologyNodeClick}><Background color="#334155" gap={18} /><Controls /></ReactFlow></ReactFlowProvider>
                    </div>
                    <h3>VMs currently recorded on this host ({hostVms.length})</h3>
                    <DataTable rows={hostVms} resetKey={selectedHostId} columns={[
                      { key: 'source_vm_id', label: 'VM', render: (row) => <button className="text-button" onClick={() => { setSelectedVmId(String(row.source_vm_id)); setPage('prediction'); }}>{String(row.source_vm_id)}</button> },
                      { key: 'provisioned_ram_mb', label: 'Provisioned RAM (MB)' },
                      { key: 'risk_alert_samples', label: 'Risk-alert samples' },
                      { key: 'sla_overload_intervals', label: 'Overload proxy intervals' },
                    ]} emptyTitle="No VM placement records" emptyDetail="The selected policy has no placement rows for this host." />
                    <h3>Migration history touching host {selectedHost.host_id}</h3>
                    <DataTable rows={hostMigrations} resetKey={selectedHostId} columns={[
                      { key: 'simulation_time', label: 'Simulation time' },
                      { key: 'source_vm_id', label: 'VM' },
                      { key: 'source_host', label: 'Source' },
                      { key: 'target_host', label: 'Destination' },
                      { key: 'event', label: 'Outcome', render: (row) => (
                        <span className="button-row">
                          <span className={eventClass(String(row.event))}>{String(row.event)}</span>
                          {row.event === 'SUCCEEDED' && (
                            <button
                              className="text-button"
                              onClick={() => setReplayEventKey(`${String(row.simulation_time)}:${String(row.source_vm_id)}:${String(row.source_host)}:${String(row.target_host)}`)}
                            >
                              Replay confirmed move
                            </button>
                          )}
                        </span>
                      ) },
                      { key: 'details', label: 'Feasibility / details' },
                    ]} emptyTitle="No migration events" emptyDetail="No recorded migration event involves this host." />
                  </>
                ) : <EmptyState title="Host details unavailable" detail="No host utilization records are available for the selected policy." />}
              </div>
            </div>
          </section>
        </>
      );
    }

    if (page === 'simulation') {
      const active = selectedRun;
      const runSummary = active?.status === 'COMPLETED'
        && active.policy === selectedPolicy
        && active.dataset_id === activeDatasetId
        && active.run_id === policyRunId
        ? policySummary
        : null;
      const selectedInput = datasetInfo?.datasets.find((dataset) => dataset.dataset_id === activeDatasetId);
      return (
        <>
          <section className="panel section-panel">
            <div className="section-header"><h2>Simulation Control Room</h2><span className="badge warning">CloudSim run</span></div>
            <div className="control-row">
              <label className="control-group"><span>Primary results policy</span><select value={runPolicy} onChange={(event) => setRunPolicy(event.target.value)}>{POLICIES.map((policy) => <option key={policy}>{policy}</option>)}</select></label>
              <div className="control-group grow"><span>Input dataset</span><strong>{selectedInput?.filename ?? 'No active upload'} · {activeDatasetId ?? '—'}</strong></div>
              <label className="control-group"><span>VM IDs (comma-separated; ALL = all)</span><input value={runVmId} onChange={(event) => setRunVmId(event.target.value)} placeholder="ALL or 1, 10, 100" /></label>
              <button className="button primary" disabled={busy || !activeDatasetId || !['upload', 'local'].includes(selectedInput?.source ?? '') || selectedInput?.dataset_kind === 'incompatible_raw_workload_csv' || (selectedInput?.dataset_kind === 'prediction_csv' && selectedInput.validation.simulator_compatible === false) || Boolean(runs.find((run) => ['PENDING', 'RUNNING'].includes(run.status)))} onClick={() => void startRun()}>{busy ? 'Submitting…' : 'Run Pipeline'}</button>
              <button className="button secondary" onClick={resetRunView}>Reset view</button>
            </div>
            <div className="toolbar">
              <label className="control-group"><span>Window start</span><input type="datetime-local" value={runStartTime} onChange={(event) => setRunStartTime(event.target.value)} /></label>
              <label className="control-group"><span>Window end</span><input type="datetime-local" value={runEndTime} onChange={(event) => setRunEndTime(event.target.value)} /></label>
              <label className="control-group"><span>Risk threshold</span><input type="number" min="0" max="1" step="0.01" value={runRiskThreshold} onChange={(event) => setRunRiskThreshold(event.target.value)} /></label>
              <label className="control-group"><span>Host count</span><input type="number" min="10" max="500" step="1" value={runHostCount} onChange={(event) => setRunHostCount(event.target.value)} /></label>
              <label className="control-group"><span>Host RAM (MB)</span><input type="number" min="1024" max="2097152" step="1024" value={runHostRamMb} onChange={(event) => setRunHostRamMb(event.target.value)} /></label>
              <label className="control-group"><span>Host CPU cores (PEs)</span><input type="number" min="4" max="256" step="4" value={runHostPes} onChange={(event) => setRunHostPes(event.target.value)} /></label>
              <label className="control-group"><span>CPU overload limit</span><input type="number" min="0.1" max="1" step="0.01" value={runCpuLimit} onChange={(event) => setRunCpuLimit(event.target.value)} /></label>
              <label className="control-group"><span>RAM overload limit</span><input type="number" min="0.1" max="1" step="0.01" value={runRamLimit} onChange={(event) => setRunRamLimit(event.target.value)} /></label>
            </div>
            {selectedInput?.validation.simulator_compatible === false && (
              <div className="alert error"><strong>Simulator input incompatibility</strong>{(selectedInput.validation.simulator_errors ?? []).map((item) => <span key={item}>{item}</span>)}</div>
            )}
          </section>
          <section className="content-grid two-col">
            <article className="panel">
              <div className="section-header compact"><h3>Run status</h3><button className="button secondary small" onClick={() => void refreshRuns()}>Refresh</button></div>
              {active ? (
                <div className="run-status-card">
                  <div className="status-line"><strong>{active.status}</strong><span>{active.stage}</span></div>
                  <dl className="detail-grid">
                    <div><dt>Run ID</dt><dd>{active.run_id}</dd></div>
                    <div><dt>Policy</dt><dd>{active.policy}</dd></div>
                    <div><dt>Dataset</dt><dd>{active.dataset_id}</dd></div>
                    <div><dt>Started</dt><dd>{active.started_at ?? 'Queued'}</dd></div>
                    <div><dt>Input mode</dt><dd>{active.data_mode === 'raw_workload_csv' ? active.inference_mode === 'resumed_predictions' ? 'Resumed model predictions' : active.inference_mode === 'fresh_and_resumed_predictions' ? 'Fresh + resumed model predictions' : 'Fresh model inference' : 'Recorded predictions'}</dd></div>
                    {active.data_mode === 'raw_workload_csv' && <div><dt>Inference VMs processed</dt><dd>{active.inference_completed_vm_count?.toLocaleString() ?? '0'} / {active.inference_total_vm_count?.toLocaleString() ?? 'Pending'} in this selection</dd></div>}
                    <div><dt>Selected rows / VMs</dt><dd>{active.selected_prediction_rows?.toLocaleString() ?? 'Pending'} / {active.selected_prediction_vms?.toLocaleString() ?? 'Pending'}</dd></div>
                    <div><dt>Simulation threshold</dt><dd>{active.run_parameters?.risk_threshold ?? '—'}</dd></div>
                    <div><dt>Host assumptions</dt><dd>{active.run_parameters ? `${active.run_parameters.host_count} hosts · ${active.run_parameters.host_ram_mb.toLocaleString()} MB · ${active.run_parameters.host_pes} PEs` : '—'}</dd></div>
                  </dl>
                  {typeof active.progress === 'number' ? <progress value={active.progress} max={100} /> : <p className="muted">Stage: {active.stage}</p>}
                  {active.error && <div className="alert error">{active.error}</div>}
                </div>
              ) : <EmptyState title="No run selected" detail="Start a simulation or select a record from the run history below." />}
            </article>
            <article className="panel">
              <h3>Completed run result summary</h3>
              {runSummary ? (
                <div className="summary-list">
                  {['migration_requests', 'successful_migrations', 'failed_migrations', 'incomplete_migrations', 'host_cpu_overload_snapshots', 'host_ram_overload_snapshots', 'energy_kwh'].map((key) => (
                    <div className="summary-row" key={key}><span>{key.replace(/_/g, ' ')}</span><strong>{runSummary[key] ?? 'Not reported'}</strong></div>
                  ))}
                </div>
              ) : <EmptyState title="No matching completed output selected" detail="Only a verified completed run for the active dataset and selected policy can populate this summary." />}
            </article>
          </section>
          <details className="methodology-details">
            <summary>Technical details · process logs</summary>
            <div className="content-grid two-col">
              <label className="log-panel"><span>stdout</span><pre>{runLogs.stdout || 'No stdout captured yet.'}</pre></label>
              <label className="log-panel"><span>stderr</span><pre>{runLogs.stderr || 'No stderr captured yet.'}</pre></label>
            </div>
          </details>
          <section className="panel">
            <h3>Run history</h3>
            <DataTable rows={runs.map(({ run_parameters: _runParameters, ...row }) => row)} columns={[
              { key: 'run_id', label: 'Run ID', render: (row) => <button className="text-button" onClick={() => selectRun(String(row.run_id))}>{String(row.run_id)}</button> },
              { key: 'policy', label: 'Policy' },
              { key: 'dataset_id', label: 'Dataset ID', render: (row) => {
                const dataset = datasetInfo?.datasets.find((item) => item.dataset_id === String(row.dataset_id));
                return `${dataset?.filename ?? 'Dataset no longer listed'} · ${String(row.dataset_id)}`;
              } },
              { key: 'status', label: 'Status' },
              { key: 'stage', label: 'Stage' },
              { key: 'created_at', label: 'Created' },
            ]} emptyTitle="No simulation runs yet" emptyDetail="A run record will appear here after you submit a simulation." />
          </section>
        </>
      );
    }

    if (page === 'benchmark') {
      const comparisonExportQuery = query({
        dataset_id: activeDatasetId,
        source: 'fresh',
        run_id: policyRunId,
        search: benchmarkSearch.trim() || undefined,
        sort_by: sortMetric,
        sort_order: sortDirection,
      });
      const availablePolicies = new Set(comparison.map((row) => row.policy));
      const pendingPolicies = POLICIES.filter((policy) => !availablePolicies.has(policy));
      return (
        <>
      <section className="panel section-panel">
        <div className="section-header"><div><div className="eyebrow">Source: {resultSourceLabel}</div><h2>Policy Benchmark</h2></div>{hasDatasetResults ? <a className="button secondary" href={`${API}/export/policy_comparison${comparisonExportQuery}`}>Download comparison CSV</a> : <button className="button secondary" disabled title="A matching completed simulation is required before exporting.">Download comparison CSV</button>}</div>
        {!hasDatasetResults && <EmptyState title="No matching simulation results yet" detail="Policy benchmark metrics are unavailable until a simulation completes for the selected dataset." />}
        {hasDatasetResults && pendingPolicies.length > 0 && <div className="alert warning">Available policy outputs: {Array.from(availablePolicies).join(', ') || 'none'}. Pending or unavailable: {pendingPolicies.join(', ')}. No missing metrics are fabricated.</div>}
            <div className="toolbar">
              <label className="control-group"><span>Chart metric</span><select value={comparisonMetric} onChange={(event) => setComparisonMetric(event.target.value as keyof ComparisonRow)}>{[
                ['energy_kwh', 'Approx. energy (kWh)'],
                ['successful_migrations', 'Successful migrations'],
                ['failed_or_incomplete_migrations', 'Failed / incomplete'],
                ['host_overload_snapshots', 'Host overload snapshots (proxy)'],
                ['migration_requests', 'Migration requests'],
              ].map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label className="control-group"><span>Find policy</span><input value={benchmarkSearch} onChange={(event) => setBenchmarkSearch(event.target.value)} placeholder="Search" /></label>
              <label className="control-group"><span>Sort table by</span><select value={sortMetric} onChange={(event) => setSortMetric(event.target.value as keyof ComparisonRow)}>{[
                'energy_kwh', 'migration_requests', 'successful_migrations', 'failed_or_incomplete_migrations', 'host_overload_snapshots',
              ].map((metric) => <option key={metric} value={metric}>{metric.replace(/_/g, ' ')}</option>)}</select></label>
              <button className="button secondary" onClick={() => setSortDirection((direction) => direction === 'asc' ? 'desc' : 'asc')}>Order: {sortDirection === 'asc' ? 'Low → high' : 'High → low'}</button>
            </div>
            <div className="chart-area">{metricChartRows.some((row) => row.value !== null) ? <ResponsiveContainer width="100%" height="100%"><BarChart data={metricChartRows}><CartesianGrid strokeDasharray="3 3" stroke="#26354a" /><XAxis dataKey="policy" stroke="#94a3b8" /><YAxis stroke="#94a3b8" /><Tooltip /><Bar dataKey="value" name={String(comparisonMetric).replace(/_/g, ' ')} fill="#38bdf8" /></BarChart></ResponsiveContainer> : <EmptyState title="Metric unavailable" detail="No completed run for this dataset reports the selected metric." />}</div>
            <div className="notice"><strong>Metric definitions</strong><span>{overview?.metric_definitions.energy_kwh} {overview?.metric_definitions.host_overload_snapshots}</span></div>
            <DataTable rows={sortedComparison} resetKey={`${benchmarkSearch}:${sortMetric}:${sortDirection}`} columns={[
              { key: 'policy', label: 'Policy' },
              { key: 'run_id', label: 'Run ID', render: (row) => row.run_id ? String(row.run_id) : 'Unavailable' },
              { key: 'energy_kwh', label: 'Approx. energy (kWh)' },
              { key: 'migration_requests', label: 'Requests' },
              { key: 'successful_migrations', label: 'Succeeded' },
              { key: 'failed_or_incomplete_migrations', label: 'Failed / incomplete' },
              { key: 'no_feasible_destination_decisions', label: 'No feasible destination' },
              { key: 'host_overload_snapshots', label: 'Host overload snapshots (proxy)' },
              { key: 'decision_rows', label: 'Decision rows' },
            ]} emptyTitle="No matching policies" emptyDetail="Change the policy search filter." />
          </section>
        </>
      );
    }

    if (page === 'migrations') {
      const resultQuery = query({
        dataset_id: activeDatasetId,
        ...(policyRunId ? { run_id: policyRunId } : {}),
      });
      return (
        <>
          <section className="panel section-panel">
            <div className="section-header"><div><div className="eyebrow">Source: {resultSourceLabel}</div><h2>Migration Explorer & Decision Ledger</h2></div><div className="button-row">{hasMatchingResults ? <><a className="button secondary" href={`${API}/export/${selectedPolicy}/migrations${resultQuery}`}>Export events CSV</a><a className="button secondary" href={`${API}/export/${selectedPolicy}/decisions${resultQuery}`}>Export decisions CSV</a></> : <><button className="button secondary" disabled title="A matching completed simulation is required before exporting.">Export events CSV</button><button className="button secondary" disabled title="A matching completed simulation is required before exporting.">Export decisions CSV</button></>}</div></div>
            {!hasMatchingResults && <EmptyState title="No matching simulation results yet" detail="Run a simulation on this dataset to populate its migration events and decision ledger." />}
            <div className="toolbar">
              <label className="control-group"><span>Policy</span><select value={selectedPolicy} onChange={(event) => { setSelectedPolicy(event.target.value); setMigrationPage(1); setDecisionPage(1); }} >{POLICIES.map((policy) => <option key={policy}>{policy}</option>)}</select></label>
              <label className="control-group"><span>Event status</span><select value={eventFilter} onChange={(event) => { setEventFilter(event.target.value); setMigrationPage(1); }}><option value="">All events</option>{EVENTS.map((event) => <option key={event}>{event}</option>)}</select></label>
              <label className="control-group"><span>Host</span><select value={migrationHostFilter} onChange={(event) => { setMigrationHostFilter(event.target.value); setMigrationPage(1); }}><option value="">All hosts</option>{hosts.map((host) => <option key={host.host_id} value={host.host_id}>{host.host_id}</option>)}</select></label>
              <label className="control-group grow"><span>Search events</span><input value={migrationSearch} onChange={(event) => { setMigrationSearch(event.target.value); setMigrationPage(1); }} placeholder="VM, host, timestamp, or details" /></label>
              <label className="control-group"><span>Sort events by</span><select value={migrationSortBy} onChange={(event) => { setMigrationSortBy(event.target.value); setMigrationPage(1); }}><option value="simulation_time">Simulation time</option><option value="source_vm_id">VM</option><option value="source_host">Source host</option><option value="target_host">Destination host</option><option value="event">Event</option></select></label>
              <button className="button secondary" onClick={() => setMigrationSortOrder((order) => order === 'asc' ? 'desc' : 'asc')}>Order: {migrationSortOrder}</button>
            </div>
            <h3>Migration events</h3>
            <DataTable rows={migrationEvents} paginate={false} columns={[
              { key: 'simulation_time', label: 'Simulation time' },
              { key: 'source_vm_id', label: 'VM' },
              { key: 'source_host', label: 'Source host' },
              { key: 'target_host', label: 'Destination host' },
              { key: 'event', label: 'Event', render: (row) => <span className={eventClass(String(row.event))}>{String(row.event)}</span> },
              { key: 'details', label: 'Feasibility / details' },
            ]} emptyTitle="No migration events match" emptyDetail="Change filters or select a policy with recorded events." />
            <Pagination
              page={migrationPage}
              total={migrationTotal}
              pageSize={migrationPageSize}
              onPage={setMigrationPage}
              onPageSize={(size) => { setMigrationPageSize(size); setMigrationPage(1); }}
            />
          </section>
          <section className="panel section-panel">
            <div className="section-header"><h3>Decision ledger</h3></div>
            <div className="toolbar">
              <label className="control-group"><span>Decision</span><select value={decisionFilter} onChange={(event) => { setDecisionFilter(event.target.value); setDecisionPage(1); }}><option value="ACTIONABLE">Actionable · hide KEEP</option><option value="">All decisions</option><option value="KEEP">KEEP</option><option value="MIGRATION_REQUESTED">MIGRATION_REQUESTED</option><option value="MIGRATION_DEFERRED">MIGRATION_DEFERRED</option><option value="NO_FEASIBLE_HOST">NO_FEASIBLE_HOST</option></select></label>
              <label className="control-group"><span>Host</span><select value={decisionHostFilter} onChange={(event) => { setDecisionHostFilter(event.target.value); setDecisionPage(1); }}><option value="">All hosts</option>{hosts.map((host) => <option key={host.host_id} value={host.host_id}>{host.host_id}</option>)}</select></label>
              <label className="control-group grow"><span>Search decisions</span><input value={decisionSearch} onChange={(event) => { setDecisionSearch(event.target.value); setDecisionPage(1); }} placeholder="VM, host, timestamp, or reason" /></label>
              <label className="control-group"><span>Sort decisions by</span><select value={decisionSortBy} onChange={(event) => { setDecisionSortBy(event.target.value); setDecisionPage(1); }}><option value="timestamp">Timestamp</option><option value="vm_id">VM</option><option value="risk_probability">Risk probability</option><option value="risk_threshold">Risk threshold</option><option value="decision">Decision</option><option value="current_host">Current host</option><option value="proposed_host">Proposed host</option><option value="reason">Reason</option></select></label>
              <button className="button secondary" onClick={() => setDecisionSortOrder((order) => order === 'asc' ? 'desc' : 'asc')}>Order: {decisionSortOrder}</button>
            </div>
            <DataTable rows={decisions} paginate={false} columns={[
              { key: 'timestamp', label: 'Timestamp' },
              { key: 'vm_id', label: 'VM' },
              { key: 'risk_probability', label: 'Risk', render: (row) => display(row.risk_probability, 4) },
              { key: 'risk_threshold', label: 'Threshold' },
              { key: 'decision', label: 'Decision' },
              { key: 'current_host', label: 'Current host' },
              { key: 'proposed_host', label: 'Proposed host' },
              { key: 'reason', label: 'Recorded reason' },
            ]} emptyTitle="No decisions match" emptyDetail="Change filters or select a policy with decision records." />
            <Pagination
              page={decisionPage}
              total={decisionTotal}
              pageSize={decisionPageSize}
              onPage={setDecisionPage}
              onPageSize={(size) => { setDecisionPageSize(size); setDecisionPage(1); }}
            />
          </section>
        </>
      );
    }

    return (
      <>
        <section className="panel section-panel">
          <div className="section-header"><div><div className="eyebrow">Pipeline · Dataset stage</div><h2>Dataset upload</h2></div></div>
          <p className="muted">{uploadFile?.name ?? 'No CSV selected.'}</p>
          <div className="button-row">
            <label className="button secondary file-button">Choose CSV<input type="file" accept=".csv,text/csv" onChange={(event) => chooseUploadFile(event.target.files?.[0] ?? null)} /></label>
            <button className="button secondary" disabled={!uploadFile || busy || !uploadFile.name.toLowerCase().endsWith('.csv') || uploadFile.size > (datasetInfo?.upload_max_bytes ?? 200 * 1024 * 1024)} onClick={() => void validateDataset()}>{busy ? 'Uploading & validating…' : 'Upload & Validate'}</button>
            <button className="button secondary" disabled={busy} onClick={() => void validateLocalDataset()}>{busy ? 'Validating local CSV…' : 'Use configured local CSV'}</button>
          </div>
          {datasetInfo?.datasets.length ? (
            <div className="control-row saved-dataset-picker">
              <label className="control-group grow">
                <span>Saved dataset</span>
                <select value={savedDatasetId || activeDatasetId || ''} onChange={(event) => setSavedDatasetId(event.target.value)}>
                  {datasetInfo.datasets.map((dataset) => (
                    <option key={dataset.dataset_id} value={dataset.dataset_id}>
                      {dataset.filename} · {dataset.vm_count.toLocaleString()} VMs · {dataset.source}
                    </option>
                  ))}
                </select>
              </label>
              <button className="button secondary" disabled={busy || !savedDatasetId || savedDatasetId === activeDatasetId} onClick={() => void switchSavedDataset()}>
                {busy ? 'Switching…' : 'Open dataset'}
              </button>
            </div>
          ) : null}
          {uploadValidation && (
            <section className="panel section-panel">
              <div className="section-header"><div><h3>{uploadValidation.filename}</h3><p className="muted">{uploadValidation.validation.row_count.toLocaleString()} rows · {uploadValidation.validation.vm_count.toLocaleString()} VMs</p></div><span className={`badge ${uploadValidation.validation.valid ? 'success' : 'danger'}`}>{uploadValidation.validation.valid ? 'Valid' : 'Invalid'}</span></div>
              {uploadValidation.validation.errors.map((item) => <div className="alert error" key={item}>{item}</div>)}
              {uploadValidation.validation.warnings.slice(0, 2).map((item) => <p className="muted" key={item}>{item}</p>)}
              <div className="button-row">
                <button className="button primary" disabled={!uploadValidation.validation.valid || !uploadValidation.dataset_id || busy} onClick={() => void storeDataset()}>{busy ? 'Working…' : 'Run Pipeline'}</button>
              </div>
            </section>
          )}
        </section>
      </>
    );
  };

  const activeTitle = sectionTitle;

  if (page === 'dataset' && !activeDatasetId) {
    return (
      <main className="initial-upload-shell">
        <section className="initial-upload">
          <div className="brand-badge">PX</div>
          <div className="eyebrow">Patch-based predictive VM consolidation</div>
          <h1>PatchXProCon</h1>
          <p className="muted">A research pipeline for VM workload forecasting and CloudSim-based consolidation.</p>
          <div className="button-row">
            <label className="button secondary file-button">Choose CSV<input type="file" accept=".csv,text/csv" onChange={(event) => chooseUploadFile(event.target.files?.[0] ?? null)} /></label>
            <button className="button primary" disabled={!uploadFile || busy || !uploadFile.name.toLowerCase().endsWith('.csv') || uploadFile.size > (datasetInfo?.upload_max_bytes ?? 200 * 1024 * 1024)} onClick={() => void validateDataset()}>{busy ? 'Uploading & validating…' : 'Upload & Validate'}</button>
            <button className="button secondary" disabled={busy} onClick={() => void validateLocalDataset()}>{busy ? 'Validating local CSV…' : 'Use configured local CSV'}</button>
          </div>
          {uploadFile && <p className="muted">{uploadFile.name}</p>}
          {error && <div className="alert error" role="alert">{error}</div>}
          {success && <div className="alert success" role="status">{success}</div>}
          {uploadValidation && (
            <section className="upload-validation">
              <div className="section-header">
                <div><h2>{uploadValidation.filename}</h2><p className="muted">{uploadValidation.validation.row_count.toLocaleString()} rows · {uploadValidation.validation.vm_count.toLocaleString()} VMs</p></div>
                <span className={`badge ${uploadValidation.validation.valid ? 'success' : 'danger'}`}>{uploadValidation.validation.valid ? 'Valid' : 'Invalid'}</span>
              </div>
              {uploadValidation.validation.errors.map((item) => <div className="alert error" key={item}>{item}</div>)}
              <button className="button primary" disabled={!uploadValidation.validation.valid || !uploadValidation.dataset_id || busy} onClick={() => void storeDataset()}>{busy ? 'Working…' : 'Run Pipeline'}</button>
            </section>
          )}
        </section>
        {confirmation && (
          <ConfirmationDialogView
            confirmation={confirmation}
            busy={busy}
            onCancel={() => setConfirmation(null)}
            onConfirm={confirmPendingAction}
          />
        )}
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-wrap">
          <div className="brand-badge">PX</div>
          <div className="brand-name">PatchXProCon</div>
        </div>
        <nav className="nav" aria-label="Dashboard pages">
          {NAV.map((item) => (
            <button key={item.key} className={`nav-item ${activeSection === item.key ? 'active' : ''}`} aria-current={activeSection === item.key ? 'page' : undefined} onClick={() => { setPage(item.page); setError(''); }}>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <div className="eyebrow">Patch-based predictive VM consolidation</div>
            <h1>{activeTitle}</h1>
          </div>
          <div className="topbar-actions">
            {health === 'offline' && <span className="status-pill offline"><i />API offline</span>}
            <button className="button secondary small" onClick={() => void refreshAll()} disabled={loading}>Refresh data</button>
          </div>
        </header>
        {workflowSteps.length > 0 && (
          <nav className="workflow-nav" aria-label={activeSection === 'pipeline' ? 'Pipeline stages' : 'Results views'}>
            {workflowSteps.map((step) => (
              <button
                key={step.key}
                className={`workflow-step ${page === step.key ? 'active' : ''}`}
                aria-current={page === step.key ? 'step' : undefined}
                onClick={() => { setPage(step.key); setError(''); }}
              >
                <strong>{step.label}</strong>
              </button>
            ))}
          </nav>
        )}
        {error && <div className="alert error" role="alert"><strong>Request failed</strong><span>{error}</span><button className="alert-close" onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
        {success && <div className="alert success" role="status"><span>{success}</span><button className="alert-close" onClick={() => setSuccess('')} aria-label="Dismiss success">×</button></div>}
        {confirmation && (
          <ConfirmationDialogView
            confirmation={confirmation}
            busy={busy}
            onCancel={() => setConfirmation(null)}
            onConfirm={confirmPendingAction}
          />
        )}
        {loading ? <div className="page-loading"><LoadingRows count={7} /><LoadingRows count={5} /></div> : renderPage()}
      </main>
    </div>
  );
}

export default App;
