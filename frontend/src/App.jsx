
import { useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BarChart3,
  Bell, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Clock3,
  Database, FileSearch, Fingerprint, LayoutDashboard, Menu, Monitor,
  Search, Settings, Shield, ShieldAlert, ShieldCheck, Users, X,
  RefreshCw, CalendarDays, SlidersHorizontal, LockKeyhole, CheckCircle2,
  Server, Laptop, FileWarning, BrainCircuit
} from 'lucide-react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, PieChart, Pie, Cell
} from 'recharts'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const navigation = [
  {
    label: 'WORKSPACE',
    items: [
      { name: 'Overview', icon: LayoutDashboard },
      { name: 'Employees', icon: Users },
      { name: 'Threat Detection', icon: ShieldAlert },
      { name: 'Investigations', icon: FileSearch },
    ],
  },
  {
    label: 'INTELLIGENCE',
    items: [
      { name: 'Risk Analytics', icon: BarChart3 },
      { name: 'Behavioural Patterns', icon: Activity },
    ],
  },
  {
    label: 'SYSTEM',
    items: [
      { name: 'Dataset Management', icon: Database },
      { name: 'Model Management', icon: BrainCircuit },
      { name: 'Settings', icon: Settings },
    ],
  },
]

const riskColors = {
  LOW: '#16a34a',
  MODERATE: '#d97706',
  HIGH: '#ea580c',
  CRITICAL: '#dc2626',
  UNKNOWN: '#94a3b8',
}

function getRiskColor(level) {
  return riskColors[String(level || 'UNKNOWN').toUpperCase()] || riskColors.UNKNOWN
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString()
}

function getDetectors(result) {
  const detectors = result?.detectors
  if (Array.isArray(detectors)) return detectors
  if (detectors && typeof detectors === 'object') {
    return Object.entries(detectors).map(([key, value]) => ({
      detector: key,
      ...value,
    }))
  }
  return []
}

function getDetectorName(detector) {
  return detector.detector_name || detector.detector || detector.name || 'Detector'
}

function getDetectorStatus(detector) {
  return String(
    detector.status || detector.result || detector.classification || 'UNKNOWN'
  ).toUpperCase()
}

function getDetectorSeverity(detector) {
  return String(detector.severity || detector.risk_level || 'UNKNOWN').toUpperCase()
}

function App() {
  const [page, setPage] = useState('Overview')
  const [health, setHealth] = useState(null)
  const [employees, setEmployees] = useState([])
  const [employeeSearch, setEmployeeSearch] = useState('')
  const [employee, setEmployee] = useState('')
  const [targetDate, setTargetDate] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [loadingHealth, setLoadingHealth] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  async function loadSystemData() {
    setLoadingHealth(true)
    setError('')

    try {
      const [healthResponse, employeeResponse] = await Promise.all([
        fetch(`${API}/api/health`),
        fetch(`${API}/api/employees?limit=1000&offset=0`),
      ])

      if (!healthResponse.ok) throw new Error('Could not load system health.')
      if (!employeeResponse.ok) throw new Error('Could not load employees.')

      const healthData = await healthResponse.json()
      const employeeData = await employeeResponse.json()

      setHealth(healthData)

      const employeeList = Array.isArray(employeeData)
        ? employeeData
        : employeeData.employees || employeeData.data || []

      setEmployees(employeeList)
    } catch (err) {
      setError(`${err.message} Check that the A.E.G.I.S. backend is running.`)
    } finally {
      setLoadingHealth(false)
    }
  }

  useEffect(() => {
    loadSystemData()
  }, [])

  const filteredEmployees = useMemo(() => {
    const query = employeeSearch.trim().toLowerCase()
    if (!query) return employees

    return employees.filter((item) => {
      const id = typeof item === 'string'
        ? item
        : item.employee || item.user || item.employee_id || item.id || ''

      return String(id).toLowerCase().includes(query)
    })
  }, [employees, employeeSearch])

  const detectorResults = useMemo(() => getDetectors(analysis), [analysis])

  const detectorChartData = useMemo(() => {
    return detectorResults.map((detector) => ({
      name: getDetectorName(detector).replaceAll('_', ' '),
      score: Number(detector.score ?? detector.risk_score ?? 0),
      status: getDetectorStatus(detector),
    }))
  }, [detectorResults])

  const assessment = analysis?.assessment || {}
  const riskLevel = String(
    assessment.risk_level || analysis?.risk_level || 'UNKNOWN'
  ).toUpperCase()
  const riskScore = Number(
    assessment.risk_score ?? analysis?.risk_score ?? 0
  )

  const anomalyCount = detectorResults.filter((item) =>
    getDetectorStatus(item).includes('ANOMALOUS')
  ).length

  const normalCount = detectorResults.filter((item) =>
    getDetectorStatus(item).includes('NORMAL')
  ).length

  const insufficientCount = detectorResults.filter((item) =>
    getDetectorStatus(item).includes('INSUFFICIENT')
  ).length

  async function runAnalysis(event) {
    event.preventDefault()

    if (!employee || !targetDate) {
      setError('Select an employee and target date first.')
      return
    }

    setLoading(true)
    setError('')
    setAnalysis(null)

    try {
      const response = await fetch(`${API}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          employee,
          target_date: targetDate,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Analysis request failed.')
      }

      setAnalysis(data)
      setPage('Threat Detection')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function selectPage(name) {
    setPage(name)
    setSidebarOpen(false)
  }

  const systemOnline = health?.status === 'online'

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button
          className="mobile-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation"
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark">
            <Shield size={22} strokeWidth={2.2} />
          </div>
          <div>
            <div className="brand-name">A.E.G.I.S.</div>
            <div className="brand-caption">SECURITY INTELLIGENCE</div>
          </div>
          <button
            className="mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={19} />
          </button>
        </div>

        <div className="workspace-switcher">
          <div className="workspace-icon"><Fingerprint size={18} /></div>
          <div className="workspace-text">
            <span>Docspace</span>
            <strong>Security Operations</strong>
          </div>
          <ChevronDown size={15} className="muted-icon" />
        </div>

        <nav className="side-navigation">
          {navigation.map((group) => (
            <div className="nav-group" key={group.label}>
              <div className="nav-label">{group.label}</div>
              {group.items.map((item) => {
                const Icon = item.icon
                return (
                  <button
                    key={item.name}
                    className={`nav-item ${page === item.name ? 'active' : ''}`}
                    onClick={() => selectPage(item.name)}
                  >
                    <Icon size={18} strokeWidth={1.8} />
                    <span>{item.name}</span>
                    {item.name === 'Threat Detection' && analysis && (
                      <span className="nav-indicator" />
                    )}
                  </button>
                )
              })}
            </div>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="help-card">
            <div className="help-icon"><CircleHelp size={17} /></div>
            <div>
              <strong>Need assistance?</strong>
              <span>View project documentation</span>
            </div>
          </div>
          <div className="profile">
            <div className="profile-avatar">A</div>
            <div className="profile-info">
              <strong>Administrator</strong>
              <span>System operator</span>
            </div>
            <Settings size={17} className="muted-icon" />
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={21} />
            </button>
            <div className="breadcrumb">
              <span>Workspace</span>
              <ChevronRight size={14} />
              <strong>{page}</strong>
            </div>
          </div>

          <div className="topbar-actions">
            <div className={`connection-status ${systemOnline ? 'online' : 'offline'}`}>
              <span className="status-dot" />
              {loadingHealth ? 'Connecting' : systemOnline ? 'System online' : 'Disconnected'}
            </div>
            <button
              className="icon-button"
              onClick={loadSystemData}
              title="Refresh data"
              aria-label="Refresh data"
            >
              <RefreshCw size={17} />
            </button>
            <button className="icon-button notification-button" aria-label="Notifications">
              <Bell size={18} />
            </button>
            <div className="topbar-avatar">A</div>
          </div>
        </header>

        <div className="page-content">
          {error && (
            <div className="error-banner">
              <AlertTriangle size={17} />
              <span>{error}</span>
              <button onClick={() => setError('')} aria-label="Dismiss error">
                <X size={16} />
              </button>
            </div>
          )}

          {page === 'Overview' && (
            <OverviewPage
              health={health}
              employees={employees}
              analysis={analysis}
              riskLevel={riskLevel}
              riskScore={riskScore}
              anomalyCount={anomalyCount}
              detectorResults={detectorResults}
              onNavigate={selectPage}
              onRefresh={loadSystemData}
              loading={loadingHealth}
            />
          )}

          {page === 'Employees' && (
            <EmployeesPage
              employees={filteredEmployees}
              search={employeeSearch}
              setSearch={setEmployeeSearch}
              setEmployee={setEmployee}
              setPage={selectPage}
            />
          )}

          {page === 'Threat Detection' && (
            <ThreatDetectionPage
              employee={employee}
              setEmployee={setEmployee}
              targetDate={targetDate}
              setTargetDate={setTargetDate}
              employees={employees}
              runAnalysis={runAnalysis}
              loading={loading}
              analysis={analysis}
              detectorResults={detectorResults}
              assessment={assessment}
              riskLevel={riskLevel}
              riskScore={riskScore}
              anomalyCount={anomalyCount}
              normalCount={normalCount}
              insufficientCount={insufficientCount}
              detectorChartData={detectorChartData}
            />
          )}

          {page === 'Investigations' && (
            <PlaceholderPage
              icon={FileSearch}
              title="Investigations"
              description="A central workspace for reviewing suspicious employee activity and documenting investigation outcomes."
              features={['Flagged activity review', 'Evidence timeline', 'Investigation notes', 'Case status tracking']}
            />
          )}

          {page === 'Risk Analytics' && (
            <PlaceholderPage
              icon={BarChart3}
              title="Risk Analytics"
              description="Historical risk trends and organization-wide insights will appear here when aggregate analytics endpoints are available."
              features={['Risk distribution', 'Risk trends over time', 'Department comparisons', 'Detector performance']}
            />
          )}

          {page === 'Behavioural Patterns' && (
            <PlaceholderPage
              icon={Activity}
              title="Behavioural Patterns"
              description="Explore employee behaviour against historical baselines and identify unusual changes in activity."
              features={['Login behaviour', 'Computer usage patterns', 'File activity patterns', 'Device activity patterns']}
            />
          )}

          {page === 'Dataset Management' && (
            <DatasetPage health={health} loading={loadingHealth} />
          )}

          {page === 'Model Management' && (
            <PlaceholderPage
              icon={BrainCircuit}
              title="Model Management"
              description="A future workspace for model configuration, versioning and evaluation."
              features={['Model versions', 'Detector configuration', 'Evaluation reports', 'Model health']}
            />
          )}

          {page === 'Settings' && (
            <PlaceholderPage
              icon={Settings}
              title="Settings"
              description="System preferences and configuration options."
              features={['Workspace preferences', 'Detection thresholds', 'Access management', 'Audit configuration']}
            />
          )}
        </div>
      </main>
    </div>
  )
}

function PageHeading({ eyebrow = 'SECURITY OVERVIEW', title, description, actions }) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="heading-actions">{actions}</div>}
    </div>
  )
}

function MetricCard({ title, value, subtitle, icon: Icon, tone = 'blue' }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <div className={`metric-icon ${tone}`}><Icon size={19} /></div>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-subtitle">{subtitle}</div>
    </div>
  )
}

function OverviewPage({
  health, employees, analysis, riskLevel, riskScore,
  anomalyCount, detectorResults, onNavigate, onRefresh, loading
}) {
  const hasAnalysis = Boolean(analysis)
  const latestEmployee = analysis?.employee || analysis?.user || '—'

  return (
    <>
      <PageHeading
        title="Overview"
        description="Monitor your security environment and employee activity."
        actions={
          <button className="secondary-button" onClick={onRefresh}>
            <RefreshCw size={16} /> Refresh
          </button>
        }
      />

      <div className="welcome-banner">
        <div className="welcome-content">
          <div className="welcome-tag"><ShieldCheck size={14} /> INSIDER THREAT MONITORING</div>
          <h2>Your security operations, <span>in focus.</span></h2>
          <p>Monitor behavioural deviations and review employee activity from one workspace.</p>
        </div>
        <div className="welcome-art">
          <div className="art-circle art-circle-one" />
          <div className="art-circle art-circle-two" />
          <div className="art-shield"><Shield size={54} strokeWidth={1.2} /></div>
          <div className="art-orbit orbit-one" />
          <div className="art-orbit orbit-two" />
        </div>
      </div>

      <div className="section-heading">
        <div>
          <h2>System snapshot</h2>
          <p>Current backend and dataset information</p>
        </div>
        <span className="live-label"><span className="status-dot" /> LIVE STATUS</span>
      </div>

      <div className="metrics-grid">
        <MetricCard
          title="System status"
          value={loading ? 'Loading' : health?.status === 'online' ? 'Online' : 'Offline'}
          subtitle="FastAPI connection"
          icon={Server}
          tone="green"
        />
        <MetricCard
          title="Employees indexed"
          value={formatNumber(health?.employee_count ?? employees.length)}
          subtitle="Available employee records"
          icon={Users}
          tone="blue"
        />
        <MetricCard
          title="Dataset records"
          value={formatNumber(health?.records)}
          subtitle="Loaded activity records"
          icon={Database}
          tone="violet"
        />
        <MetricCard
          title="Latest risk score"
          value={hasAnalysis ? `${riskScore}/100` : '—'}
          subtitle={hasAnalysis ? `Latest: ${latestEmployee}` : 'Run an assessment to populate'}
          icon={ShieldAlert}
          tone="orange"
        />
      </div>

      <div className="dashboard-grid">
        <section className="panel recent-panel">
          <div className="panel-heading">
            <div>
              <h3>Latest assessment</h3>
              <p>Most recent employee-day analysis in this session</p>
            </div>
            <button className="text-button" onClick={() => onNavigate('Threat Detection')}>
              Open detection <ChevronRight size={15} />
            </button>
          </div>

          {hasAnalysis ? (
            <div className="latest-assessment">
              <div className="assessment-employee">
                <div className="employee-avatar"><Fingerprint size={21} /></div>
                <div>
                  <strong>{latestEmployee}</strong>
                  <span>{analysis.target_date || analysis.date || 'Assessment completed'}</span>
                </div>
              </div>
              <div className="assessment-risk">
                <span className="risk-score" style={{ color: getRiskColor(riskLevel) }}>
                  {riskScore}
                </span>
                <span className="risk-level" style={{ color: getRiskColor(riskLevel) }}>
                  {riskLevel} RISK
                </span>
              </div>
              <div className="assessment-footer">
                <span>{anomalyCount} anomalous detector results</span>
                <span>{detectorResults.length} detectors evaluated</span>
              </div>
            </div>
          ) : (
            <EmptyState
              icon={ShieldCheck}
              title="No assessment yet"
              description="Run an employee-day assessment to see its risk score and detector findings here."
              action="Run threat detection"
              onClick={() => onNavigate('Threat Detection')}
            />
          )}
        </section>

        <section className="panel coverage-panel">
          <div className="panel-heading">
            <div>
              <h3>Detection coverage</h3>
              <p>Results from the latest assessment</p>
            </div>
            <div className="small-icon-box"><Activity size={17} /></div>
          </div>

          {hasAnalysis ? (
            <>
              <div className="coverage-number">{detectorResults.length}<span> detectors</span></div>
              <div className="coverage-list">
                <CoverageRow label="Anomalous" count={anomalyCount} color="#ea580c" total={detectorResults.length} />
                <CoverageRow label="Normal" count={detectorResults.filter((item) => getDetectorStatus(item).includes('NORMAL')).length} color="#16a34a" total={detectorResults.length} />
                <CoverageRow label="Insufficient data" count={detectorResults.filter((item) => getDetectorStatus(item).includes('INSUFFICIENT')).length} color="#94a3b8" total={detectorResults.length} />
              </div>
            </>
          ) : (
            <EmptyState
              icon={Activity}
              title="Awaiting detector results"
              description="Detector coverage will be calculated after an assessment."
            />
          )}
        </section>
      </div>

      <div className="section-heading lower-heading">
        <div>
          <h2>Explore your workspace</h2>
          <p>Quick access to A.E.G.I.S. modules</p>
        </div>
      </div>

      <div className="quick-grid">
        <QuickCard icon={Users} title="Employees" description="Browse indexed employee records." onClick={() => onNavigate('Employees')} />
        <QuickCard icon={ShieldAlert} title="Threat detection" description="Analyse employee activity against baselines." onClick={() => onNavigate('Threat Detection')} />
        <QuickCard icon={Database} title="Dataset management" description="Review dataset loading and telemetry." onClick={() => onNavigate('Dataset Management')} />
      </div>
    </>
  )
}

function CoverageRow({ label, count, color, total }) {
  const percentage = total ? (count / total) * 100 : 0

  return (
    <div className="coverage-row">
      <div className="coverage-row-top">
        <span><i style={{ background: color }} />{label}</span>
        <strong>{count}</strong>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percentage}%`, background: color }} />
      </div>
    </div>
  )
}

function QuickCard({ icon: Icon, title, description, onClick }) {
  return (
    <button className="quick-card" onClick={onClick}>
      <div className="quick-icon"><Icon size={20} /></div>
      <div>
        <strong>{title}</strong>
        <span>{description}</span>
      </div>
      <ChevronRight size={17} className="quick-arrow" />
    </button>
  )
}

function EmptyState({ icon: Icon, title, description, action, onClick }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><Icon size={23} /></div>
      <strong>{title}</strong>
      <p>{description}</p>
      {action && (
        <button className="primary-button small-button" onClick={onClick}>
          {action} <ChevronRight size={15} />
        </button>
      )}
    </div>
  )
}

function EmployeesPage({ employees, search, setSearch, setEmployee, setPage }) {
  function chooseEmployee(item) {
    const id = typeof item === 'string'
      ? item
      : item.employee || item.user || item.employee_id || item.id

    setEmployee(String(id))
    setPage('Threat Detection')
  }

  return (
    <>
      <PageHeading
        eyebrow="WORKSPACE"
        title="Employees"
        description="Browse employee records available in the loaded dataset."
      />

      <section className="panel">
        <div className="table-toolbar">
          <div>
            <h3>Employee directory</h3>
            <p>{employees.length.toLocaleString()} records in current result</p>
          </div>
          <div className="search-field">
            <Search size={17} />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search employee ID..."
            />
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>EMPLOYEE ID</th>
                <th>RECORD STATUS</th>
                <th>AVAILABLE ACTION</th>
              </tr>
            </thead>
            <tbody>
              {employees.map((item, index) => {
                const id = typeof item === 'string'
                  ? item
                  : item.employee || item.user || item.employee_id || item.id || `Record ${index + 1}`

                return (
                  <tr key={`${id}-${index}`}>
                    <td>
                      <div className="table-employee">
                        <div className="table-avatar"><Fingerprint size={17} /></div>
                        <strong>{id}</strong>
                      </div>
                    </td>
                    <td><span className="status-pill neutral">Indexed</span></td>
                    <td>
                      <button className="text-button" onClick={() => chooseEmployee(item)}>
                        Analyse employee <ChevronRight size={15} />
                      </button>
                    </td>
                  </tr>
                )
              })}
              {employees.length === 0 && (
                <tr><td colSpan="3" className="table-empty">No matching employees found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}

function ThreatDetectionPage({
  employee, setEmployee, targetDate, setTargetDate, employees,
  runAnalysis, loading, analysis, detectorResults, assessment,
  riskLevel, riskScore, anomalyCount, normalCount,
  insufficientCount, detectorChartData
}) {
  const [showDetails, setShowDetails] = useState(true)

  return (
    <>
      <PageHeading
        eyebrow="DETECTION ENGINE"
        title="Threat Detection"
        description="Evaluate employee activity against their historical behavioural baseline."
      />

      <section className="panel analysis-form-panel">
        <div className="panel-heading">
          <div>
            <h3>New assessment</h3>
            <p>Select an employee and date to evaluate their activity.</p>
          </div>
          <div className="small-icon-box"><SlidersHorizontal size={17} /></div>
        </div>

        <form className="analysis-form" onSubmit={runAnalysis}>
          <label>
            Employee ID
            <div className="input-with-icon">
              <Users size={17} />
              <input
                list="employee-options"
                value={employee}
                onChange={(event) => setEmployee(event.target.value)}
                placeholder="e.g. ACM2278"
                required
              />
              <datalist id="employee-options">
                {employees.map((item, index) => {
                  const id = typeof item === 'string'
                    ? item
                    : item.employee || item.user || item.employee_id || item.id

                  return <option value={id} key={`${id}-${index}`} />
                })}
              </datalist>
            </div>
          </label>

          <label>
            Target date
            <div className="input-with-icon">
              <CalendarDays size={17} />
              <input
                type="date"
                value={targetDate}
                onChange={(event) => setTargetDate(event.target.value)}
                required
              />
            </div>
          </label>

          <button className="primary-button analyze-button" type="submit" disabled={loading}>
            {loading ? <RefreshCw size={17} className="spin" /> : <ShieldCheck size={17} />}
            {loading ? 'Analysing...' : 'Run assessment'}
          </button>
        </form>
      </section>

      {analysis ? (
        <>
          <div className="section-heading results-heading">
            <div>
              <h2>Assessment results</h2>
              <p>Employee {analysis.employee || analysis.user || employee} · {analysis.target_date || analysis.date || targetDate}</p>
            </div>
            <span className="status-pill complete">
              <CheckCircle2 size={14} /> {assessment.assessment_status || 'Completed'}
            </span>
          </div>

          <div className="metrics-grid result-metrics">
            <MetricCard title="Risk score" value={`${riskScore}/100`} subtitle="Heuristic risk score" icon={ShieldAlert} tone="orange" />
            <MetricCard title="Risk level" value={riskLevel} subtitle="Assessment classification" icon={AlertTriangle} tone="orange" />
            <MetricCard title="Anomalous" value={anomalyCount} subtitle="Detectors flagged activity" icon={FileWarning} tone="red" />
            <MetricCard title="Normal" value={normalCount} subtitle={`${insufficientCount} insufficient data`} icon={ShieldCheck} tone="green" />
          </div>

          <div className="dashboard-grid result-grid">
            <section className="panel chart-panel">
              <div className="panel-heading">
                <div>
                  <h3>Detector score contribution</h3>
                  <p>Scores returned by this assessment</p>
                </div>
                <div className="small-icon-box"><BarChart3 size={17} /></div>
              </div>

              {detectorChartData.some((item) => item.score > 0) ? (
                <div className="chart-area">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={detectorChartData} margin={{ top: 10, right: 8, left: -20, bottom: 45 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f4" />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: '#7b8494', fontSize: 10 }}
                        angle={-25}
                        textAnchor="end"
                        interval={0}
                      />
                      <YAxis tick={{ fill: '#7b8494', fontSize: 11 }} />
                      <Tooltip />
                      <Bar dataKey="score" fill="#6687e8" radius={[5, 5, 0, 0]} maxBarSize={42} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState icon={BarChart3} title="No score values returned" description="This assessment did not return detector score values for charting." />
              )}
            </section>

            <section className="panel chart-panel">
              <div className="panel-heading">
                <div>
                  <h3>Detector outcomes</h3>
                  <p>Distribution within this assessment</p>
                </div>
                <div className="small-icon-box"><Activity size={17} /></div>
              </div>

              {detectorResults.length > 0 ? (
                <>
                  <div className="donut-wrap">
                    <ResponsiveContainer width="100%" height={205}>
                      <PieChart>
                        <Pie
                          data={[
                            { name: 'Anomalous', value: anomalyCount, color: '#ea580c' },
                            { name: 'Normal', value: normalCount, color: '#16a34a' },
                            { name: 'Insufficient', value: insufficientCount, color: '#cbd5e1' },
                          ].filter((item) => item.value > 0)}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={58}
                          outerRadius={82}
                          paddingAngle={4}
                        >
                          {[
                            { name: 'Anomalous', value: anomalyCount, color: '#ea580c' },
                            { name: 'Normal', value: normalCount, color: '#16a34a' },
                            { name: 'Insufficient', value: insufficientCount, color: '#cbd5e1' },
                          ].filter((item) => item.value > 0).map((item) => (
                            <Cell key={item.name} fill={item.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="donut-center">
                      <strong>{detectorResults.length}</strong>
                      <span>DETECTORS</span>
                    </div>
                  </div>
                  <div className="chart-legend">
                    <span><i className="legend-orange" /> Anomalous <strong>{anomalyCount}</strong></span>
                    <span><i className="legend-green" /> Normal <strong>{normalCount}</strong></span>
                    <span><i className="legend-grey" /> Insufficient <strong>{insufficientCount}</strong></span>
                  </div>
                </>
              ) : (
                <EmptyState icon={Activity} title="No detector results" description="No detector outcomes were returned." />
              )}
            </section>
          </div>

          <section className="panel detector-panel">
            <div className="panel-heading">
              <div>
                <h3>Detector findings</h3>
                <p>Individual detector results and severity levels</p>
              </div>
              <button className="secondary-button compact-button" onClick={() => setShowDetails(!showDetails)}>
                {showDetails ? 'Collapse' : 'Expand'} <ChevronDown size={15} />
              </button>
            </div>

            {showDetails && (
              <div className="detector-list">
                {detectorResults.map((item, index) => {
                  const status = getDetectorStatus(item)
                  const severity = getDetectorSeverity(item)
                  const anomalous = status.includes('ANOMALOUS')

                  return (
                    <div className="detector-row" key={`${getDetectorName(item)}-${index}`}>
                      <div className={`detector-icon ${anomalous ? 'detector-warning' : 'detector-safe'}`}>
                        {anomalous ? <AlertTriangle size={18} /> : <ShieldCheck size={18} />}
                      </div>
                      <div className="detector-info">
                        <strong>{getDetectorName(item).replaceAll('_', ' ')}</strong>
                        <span>{item.message || item.description || item.reason || 'Detector evaluation completed.'}</span>
                      </div>
                      <span className={`status-pill ${anomalous ? 'warning' : status.includes('NORMAL') ? 'safe' : 'neutral'}`}>
                        {status.replaceAll('_', ' ')}
                      </span>
                      <span className="severity-text" style={{ color: getRiskColor(severity) }}>
                        {severity}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </section>

          <section className="panel note-panel">
            <div className="note-icon"><LockKeyhole size={18} /></div>
            <div>
              <strong>Interpretation note</strong>
              <p>An anomalous result indicates a deviation from the configured behavioural baseline. It is an investigation signal, not proof of malicious intent.</p>
            </div>
          </section>
        </>
      ) : (
        <section className="panel detection-empty">
          <div className="large-empty-icon"><ShieldCheck size={32} /></div>
          <h3>Ready to analyse activity</h3>
          <p>Choose an employee and target date above. A.E.G.I.S. will compare recorded activity with the employee's historical baseline.</p>
          <div className="detector-chips">
            <span><Clock3 size={14} /> Login behaviour</span>
            <span><Monitor size={14} /> Computer usage</span>
            <span><FileSearch size={14} /> File activity</span>
            <span><Laptop size={14} /> Device activity</span>
          </div>
        </section>
      )}
    </>
  )
}

function DatasetPage({ health, loading }) {
  return (
    <>
      <PageHeading
        eyebrow="SYSTEM"
        title="Dataset Management"
        description="Review the dataset currently loaded by the A.E.G.I.S. backend."
      />

      <div className="metrics-grid">
        <MetricCard title="Dataset status" value={loading ? 'Loading' : health?.dataset_loaded ? 'Loaded' : 'Unavailable'} subtitle="Backend dataset state" icon={Database} tone="green" />
        <MetricCard title="Employee records" value={formatNumber(health?.employee_count)} subtitle="Unique employees reported" icon={Users} tone="blue" />
        <MetricCard title="Activity records" value={formatNumber(health?.records)} subtitle="Rows loaded by backend" icon={Activity} tone="violet" />
        <MetricCard title="Backend status" value={health?.status || 'Unknown'} subtitle="API health response" icon={Server} tone="green" />
      </div>

      <section className="panel dataset-info">
        <div className="panel-heading">
          <div>
            <h3>Dataset information</h3>
            <p>Information returned by the current health endpoint</p>
          </div>
          <div className="small-icon-box"><Database size={17} /></div>
        </div>
        <div className="dataset-detail-row">
          <span>System</span>
          <strong>{health?.system || 'A.E.G.I.S.'}</strong>
        </div>
        <div className="dataset-detail-row">
          <span>Dataset loaded</span>
          <strong>{health?.dataset_loaded ? 'Yes' : 'No'}</strong>
        </div>
        <div className="dataset-detail-row">
          <span>Employees indexed</span>
          <strong>{formatNumber(health?.employee_count)}</strong>
        </div>
        <div className="dataset-detail-row">
          <span>Total records</span>
          <strong>{formatNumber(health?.records)}</strong>
        </div>
      </section>

      <div className="info-banner">
        <Database size={18} />
        <span>Upload, replace and dataset validation controls are not connected yet. This page currently displays backend-reported dataset information only.</span>
      </div>
    </>
  )
}

function PlaceholderPage({ icon: Icon, title, description, features }) {
  return (
    <>
      <PageHeading eyebrow="WORKSPACE MODULE" title={title} description={description} />

      <section className="panel coming-soon-panel">
        <div className="coming-icon"><Icon size={28} /></div>
        <span className="coming-label">PLANNED MODULE</span>
        <h2>{title}</h2>
        <p>This module is part of the A.E.G.I.S. workspace roadmap. Its functionality can be developed in a future phase.</p>
        <div className="planned-features">
          {features.map((feature) => (
            <div className="planned-feature" key={feature}>
              <CheckCircle2 size={16} />
              {feature}
            </div>
          ))}
        </div>
      </section>
    </>
  )
}

export default App
