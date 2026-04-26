import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Activity,
  BarChart3,
  RefreshCw,
  Lock as LockIcon,
  Cpu
} from 'lucide-react';
import {
  AreaChart,
  Area,
  ResponsiveContainer
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Transaction } from './types';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

type FraudPredictionResponse = {
  fraud_probability: number;
  prediction: 0 | 1;
  threshold: number;
  model_meta?: {
    imbalance_strategy?: string;
    roc_auc?: number;
    avg_precision?: number;
  };
};

type CreditCardRow = {
  Time: number;
  Amount: number;
  Class: 0 | 1;
} & Record<`V${number}`, number>;

const API_BASE =
  window.location.port === '3000'
    ? ''
    : 'http://localhost:3000';

const CSV_PATH = '/creditcard.csv';
const STREAM_BATCH_SIZE = 3;
const STREAM_INTERVAL_MS = 2500;
const MAX_VISIBLE_TRANSACTIONS = Infinity;
const DEBUG = true;

function debugLog(...args: any[]) {
  if (DEBUG) {
    console.log('[APP]', ...args);
  }
}

function parseCreditCardCsv(csvText: string): CreditCardRow[] {
  debugLog('parseCreditCardCsv called');
  debugLog('raw csv length =', csvText.length);

  const lines = csvText.trim().split(/\r?\n/);
  debugLog('total csv lines =', lines.length);

  if (lines.length < 2) {
    debugLog('CSV has fewer than 2 lines');
    return [];
  }

  const headers = lines[0]
    .split(',')
    .map((h) => h.trim().replace(/^"|"$/g, ''));

  debugLog('CSV headers =', headers);
  debugLog('header count =', headers.length);

  const rows: CreditCardRow[] = [];

  for (let i = 1; i < lines.length; i++) {
    const rawLine = lines[i].trim();
    if (!rawLine) continue;

    const values = rawLine
      .split(',')
      .map((v) => v.trim().replace(/^"|"$/g, ''));

    if (values.length !== headers.length) {
      if (i < 5) {
        debugLog(`Skipping line ${i + 1}: values/header mismatch`, {
          valueCount: values.length,
          headerCount: headers.length,
          rawLine,
        });
      }
      continue;
    }

    const record: Record<string, number> = {};

    headers.forEach((header, idx) => {
      const parsed = Number(values[idx]);
      record[header] = Number.isFinite(parsed) ? parsed : 0;
    });

    const row: CreditCardRow = {
      Time: record['Time'] ?? 0,
      Amount: record['Amount'] ?? 0,
      Class: (record['Class'] ?? 0) === 1 ? 1 : 0,
      V1: record['V1'] ?? 0,
      V2: record['V2'] ?? 0,
      V3: record['V3'] ?? 0,
      V4: record['V4'] ?? 0,
      V5: record['V5'] ?? 0,
      V6: record['V6'] ?? 0,
      V7: record['V7'] ?? 0,
      V8: record['V8'] ?? 0,
      V9: record['V9'] ?? 0,
      V10: record['V10'] ?? 0,
      V11: record['V11'] ?? 0,
      V12: record['V12'] ?? 0,
      V13: record['V13'] ?? 0,
      V14: record['V14'] ?? 0,
      V15: record['V15'] ?? 0,
      V16: record['V16'] ?? 0,
      V17: record['V17'] ?? 0,
      V18: record['V18'] ?? 0,
      V19: record['V19'] ?? 0,
      V20: record['V20'] ?? 0,
      V21: record['V21'] ?? 0,
      V22: record['V22'] ?? 0,
      V23: record['V23'] ?? 0,
      V24: record['V24'] ?? 0,
      V25: record['V25'] ?? 0,
      V26: record['V26'] ?? 0,
      V27: record['V27'] ?? 0,
      V28: record['V28'] ?? 0,
    };

    if (rows.length < 3) {
      debugLog(`parsed row ${rows.length + 1}`, row);
    }

    rows.push(row);
  }

  debugLog('parsed dataset rows =', rows.length);

  if (rows.length > 0) {
    debugLog('first parsed row final =', rows[0]);
    debugLog('first row amount =', rows[0].Amount, 'first row class =', rows[0].Class);
  }

  return rows;
}

function mapDatasetRowToTransaction(row: CreditCardRow, id: string): Transaction {
  const mapped: Transaction = {
    id,
    time: row.Time,
    amount: row.Amount,
    v1: row.V1,
    v2: row.V2,
    v3: row.V3,
    v4: row.V4,
    v5: row.V5,
    v6: row.V6,
    v7: row.V7,
    v8: row.V8,
    v9: row.V9,
    v10: row.V10,
    v11: row.V11,
    v12: row.V12,
    v13: row.V13,
    v14: row.V14,
    v15: row.V15,
    v16: row.V16,
    v17: row.V17,
    v18: row.V18,
    v19: row.V19,
    v20: row.V20,
    v21: row.V21,
    v22: row.V22,
    v23: row.V23,
    v24: row.V24,
    v25: row.V25,
    v26: row.V26,
    v27: row.V27,
    v28: row.V28,
    riskScore: 0,
    isFraud: row.Class === 1,
    isQuantumSecure: true,
    status: 'pending',
  };

  debugLog('mapped transaction preview', {
    id: mapped.id,
    amount: mapped.amount,
    time: mapped.time,
    isFraud: mapped.isFraud,
    v1: mapped.v1,
    v28: mapped.v28,
  });

  return mapped;
}

export default function App() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [datasetRows, setDatasetRows] = useState<CreditCardRow[]>([]);
  const [datasetStatus, setDatasetStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [datasetError, setDatasetError] = useState<string | null>(null);

  const datasetIndexRef = useRef(0);
  const streamCounterRef = useRef(1);
  const isGeneratingRef = useRef(false);

  const [riskThreshold, setRiskThreshold] = useState(45);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainProgress, setRetrainProgress] = useState(0);
  const [retrainMessage, setRetrainMessage] = useState<string | null>(null);
  const [lastRetrainTime, setLastRetrainTime] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'live' | 'metrics' | 'retraining'>('live');

  const metrics = useMemo(() => {
    const tp = transactions.filter(t => t.riskScore >= riskThreshold && t.isFraud).length;
    const fp = transactions.filter(t => t.riskScore >= riskThreshold && !t.isFraud).length;
    const tn = transactions.filter(t => t.riskScore < riskThreshold && !t.isFraud).length;
    const fn = transactions.filter(t => t.riskScore < riskThreshold && t.isFraud).length;

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      f1Score,
      precision,
      recall,
      confusionMatrix: { tp, fp, tn, fn }
    };
  }, [transactions, riskThreshold]);

  const feedbackTransactions = useMemo(
    () => transactions.filter(t => t.riskScore >= riskThreshold).slice(0, 6),
    [transactions, riskThreshold]
  );

  const totalScanned = transactions.length;
  const fraudDetected = transactions.filter(t => t.riskScore >= riskThreshold).length;
  const highRiskCount = fraudDetected;

  const systemStatus =
    highRiskCount > 10 ? 'High Alert' : highRiskCount > 0 ? 'Monitoring' : 'Secure';

  const systemStatusColor =
    highRiskCount > 10
      ? 'text-red-400'
      : highRiskCount > 0
      ? 'text-yellow-400'
      : 'text-emerald-400';

  const avgRisk =
    transactions.length > 0
      ? (transactions.reduce((sum, t) => sum + t.riskScore, 0) / transactions.length).toFixed(1)
      : '0.0';

  const quantumKeys = transactions.filter(t => t.riskScore >= riskThreshold).length;

  const liveSubtitle =
    datasetStatus === 'loading'
      ? 'Loading creditcard.csv...'
      : datasetStatus === 'error'
      ? `Dataset load failed: ${datasetError}`
      : highRiskCount > 0
      ? `${highRiskCount} suspicious transactions detected in live stream.`
      : 'No suspicious transactions detected in live stream.';

  const metricsSubtitle =
    `Precision ${(metrics.precision * 100).toFixed(1)}% • Recall ${(metrics.recall * 100).toFixed(1)}%`;

  const retrainingSubtitle =
    feedbackTransactions.length > 0
      ? `${feedbackTransactions.length} feedback vectors available for retraining.`
      : 'Awaiting enough high-risk transactions for retraining.';

  const getSecurityVariant = (riskScore: number) => {
    if (riskScore >= 75) return 'Kyber-1024';
    if (riskScore >= riskThreshold) return 'Kyber-768';
    return 'Kyber-512';
  };

  const getSecurityColor = (riskScore: number) => {
    if (riskScore >= 75) return 'text-red-400';
    if (riskScore >= riskThreshold) return 'text-yellow-400';
    return 'text-emerald-400';
  };

  const activeEncryptionProfile =
    highRiskCount > 10
      ? 'Kyber-1024 Active'
      : highRiskCount > 0
      ? 'Kyber-768 Active'
      : 'Kyber-512 Active';

  const scoreTransaction = useCallback(
    async (t: Transaction) => {
      let sum = 0;

      for (let i = 1; i <= 28; i++) {
        sum += Math.abs((t as any)[`v${i}`]);
      }

      const riskScore = Math.min(100, (sum / 50) + (t.amount / 10));

      debugLog('LOCAL risk calculated', {
        id: t.id,
        sum,
        amount: t.amount,
        riskScore
      });

      return {
        riskScore,
        isFraud: riskScore >= riskThreshold,
        status: riskScore >= riskThreshold ? 'flagged' : 'pending',
      } as Partial<Transaction>;
    },
    [riskThreshold]
  );

  const generateTransactionsFromDataset = useCallback(async () => {
    debugLog('generateTransactionsFromDataset called', {
      datasetRowsLength: datasetRows.length,
      isGenerating: isGeneratingRef.current,
      datasetIndex: datasetIndexRef.current,
    });

    if (datasetRows.length === 0 || isGeneratingRef.current) return;

    isGeneratingRef.current = true;

    try {
      const batch: Transaction[] = [];

      for (let i = 0; i < STREAM_BATCH_SIZE; i++) {
        const row = datasetRows[datasetIndexRef.current % datasetRows.length];
        const id = `TX-${String(streamCounterRef.current).padStart(6, '0')}`;

        debugLog('creating tx from dataset row', {
          sourceIndex: datasetIndexRef.current % datasetRows.length,
          id,
          amount: row.Amount,
          class: row.Class,
        });

        batch.push(mapDatasetRowToTransaction(row, id));

        datasetIndexRef.current += 1;
        streamCounterRef.current += 1;
      }

      debugLog('batch created', batch.map((t) => ({
        id: t.id,
        amount: t.amount,
        time: t.time,
        isFraud: t.isFraud,
      })));

      setTransactions(prev => {
        const next = [...batch, ...prev].slice(0, MAX_VISIBLE_TRANSACTIONS);
        debugLog('transactions updated after batch insert', {
          prevLength: prev.length,
          nextLength: next.length,
        });
        return next;
      });

      batch.forEach((t) => {
        scoreTransaction(t)
          .then((scored) => {
            debugLog('applying scored transaction', t.id, scored);
            setTransactions(prev =>
              prev.map(p => (p.id === t.id ? { ...p, ...scored } : p))
            );
          })
          .catch((err) => {
            debugLog('scoreTransaction catch', t.id, err instanceof Error ? err.message : err);
            setTransactions(prev =>
              prev.map(p =>
                p.id === t.id
                  ? {
                      ...p,
                      riskScore: 0,
                      status: 'pending'
                    }
                  : p
              )
            );
          });
      });
    } finally {
      isGeneratingRef.current = false;
      debugLog('generateTransactionsFromDataset finished');
    }
  }, [datasetRows, scoreTransaction]);

  const handleRetrain = async () => {
    debugLog('handleRetrain called', {
      feedbackTransactionsLength: feedbackTransactions.length,
    });

    if (feedbackTransactions.length === 0) {
      setRetrainMessage('No high-risk transactions available for retraining.');
      return;
    }

    try {
      setIsRetraining(true);
      setRetrainProgress(15);
      setRetrainMessage('Starting retraining...');

      const payload = feedbackTransactions.map(t => ({
        Time: Math.floor(t.time),
        Amount: t.amount,
        V1: t.v1,
        V2: t.v2,
        V3: t.v3,
        V4: t.v4,
        V5: t.v5,
        V6: t.v6,
        V7: t.v7,
        V8: t.v8,
        V9: t.v9,
        V10: t.v10,
        V11: t.v11,
        V12: t.v12,
        V13: t.v13,
        V14: t.v14,
        V15: t.v15,
        V16: t.v16,
        V17: t.v17,
        V18: t.v18,
        V19: t.v19,
        V20: t.v20,
        V21: t.v21,
        V22: t.v22,
        V23: t.v23,
        V24: t.v24,
        V25: t.v25,
        V26: t.v26,
        V27: t.v27,
        V28: t.v28,
        Class: t.riskScore >= riskThreshold ? 1 : 0
      }));

      debugLog('retrain payload preview', payload.slice(0, 2));

      setRetrainProgress(45);

      const resp = await fetch(`${API_BASE}/api/fraud/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feedback: payload }),
      });

      debugLog('retrain response status', resp.status);

      if (!resp.ok) {
        let errorMessage = 'Retrain failed';

        try {
          const errorJson = await resp.json();
          debugLog('retrain failed response json', errorJson);
          errorMessage = errorJson?.message || errorJson?.error || errorMessage;
        } catch {
          const errorText = await resp.text();
          debugLog('retrain failed response text', errorText);
          errorMessage = errorText || errorMessage;
        }

        throw new Error(errorMessage);
      }

      const json = await resp.json();
      debugLog('retrain success', json);

      setRetrainProgress(100);
      setLastRetrainTime(new Date().toLocaleTimeString());
      setRetrainMessage('Retraining completed successfully.');
    } catch (err) {
      debugLog('handleRetrain catch', err);
      setRetrainProgress(0);
      setRetrainMessage(
        `Retraining failed: ${err instanceof Error ? err.message : 'Unknown error'}`
      );
    } finally {
      setIsRetraining(false);
    }
  };

  useEffect(() => {
    const loadDataset = async () => {
      try {
        debugLog('loadDataset start', { CSV_PATH, API_BASE });
        setDatasetStatus('loading');
        setDatasetError(null);

        const resp = await fetch(`${API_BASE}${CSV_PATH}`);
        debugLog('dataset fetch status', resp.status, resp.statusText);

        if (!resp.ok) {
          throw new Error(`Could not fetch ${CSV_PATH}`);
        }

        const csvText = await resp.text();
        debugLog('dataset fetch success, first 200 chars =', csvText.slice(0, 200));

        const parsed = parseCreditCardCsv(csvText);

        if (parsed.length === 0) {
          throw new Error('CSV loaded but no valid rows were parsed.');
        }

        setDatasetRows(parsed);
        setDatasetStatus('ready');

        debugLog('dataset ready', {
          rows: parsed.length,
          firstRow: parsed[0],
        });
      } catch (error) {
        debugLog('loadDataset error', error);
        setDatasetStatus('error');
        setDatasetError(error instanceof Error ? error.message : 'Unknown dataset error');
      }
    };

    loadDataset();
  }, []);

  useEffect(() => {
    debugLog('datasetRows changed', datasetRows.length);
    if (datasetRows.length > 0) {
      debugLog('datasetRows[0]', datasetRows[0]);
    }
  }, [datasetRows]);

  useEffect(() => {
    debugLog('transactions changed', transactions.length);
    if (transactions.length > 0) {
      debugLog('transactions[0]', transactions[0]);
    }
  }, [transactions]);

  useEffect(() => {
    debugLog('stream effect triggered', {
      datasetStatus,
      datasetRowsLength: datasetRows.length,
    });

    if (datasetStatus !== 'ready' || datasetRows.length === 0) return;

    generateTransactionsFromDataset();

    const interval = setInterval(() => {
      generateTransactionsFromDataset();
    }, STREAM_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [datasetStatus, datasetRows, generateTransactionsFromDataset]);

  const filteredTransactions = transactions;

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-emerald-500/30">
      <div className="fixed left-0 top-0 h-full w-64 bg-[#0A0A0A] border-r border-white/5 p-6 z-50">
        <div className="flex items-center gap-3 mb-10">
          <div className="w-10 h-10 bg-emerald-500/10 rounded-lg flex items-center justify-center border border-emerald-500/20">
            <Shield className="w-6 h-6 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight">QUANTUM-SAFE</h1>
            <p className="text-[10px] text-zinc-500 uppercase tracking-widest font-medium">
              Integrity Monitor
            </p>
          </div>
        </div>

        <nav className="space-y-1">
          {[
            { id: 'live', label: 'Live Monitor', icon: Activity },
            { id: 'metrics', label: 'Security Metrics', icon: BarChart3 },
            { id: 'retraining', label: 'Model Retraining', icon: RefreshCw },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id as any)}
              className={cn(
                'w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all',
                activeTab === item.id
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
              )}
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="mt-10 pt-10 border-t border-white/5">
          <div className="flex items-center justify-between mb-4">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
              Risk Threshold
            </span>
            <span className="text-xs font-mono text-emerald-400">{riskThreshold}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={riskThreshold}
            onChange={(e) => setRiskThreshold(parseInt(e.target.value))}
            className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
          <p className="mt-4 text-[11px] text-zinc-600 leading-relaxed">
            Filtering transactions with risk probability output ≥ {riskThreshold}%.
          </p>
        </div>

        <div className="absolute bottom-6 left-6 right-6">
          <div className="p-4 bg-zinc-900/50 rounded-2xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Cpu className="w-3 h-3 text-emerald-400" />
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">
                System Status
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  'w-1.5 h-1.5 rounded-full animate-pulse',
                  highRiskCount > 10
                    ? 'bg-red-500'
                    : highRiskCount > 0
                    ? 'bg-yellow-500'
                    : 'bg-emerald-500'
                )}
              />
              <span className={cn('text-[11px] font-medium', systemStatusColor)}>
                {activeEncryptionProfile}
              </span>
            </div>
          </div>
        </div>
      </div>

      <main className="ml-64 p-8">
        <header className="flex items-center justify-between mb-10">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">
              {activeTab === 'live' && 'Real-time Transaction Stream'}
              {activeTab === 'metrics' && 'Security Performance Analysis'}
              {activeTab === 'retraining' && 'Human-in-the-Loop Retraining'}
            </h2>
            <p className="text-zinc-500 text-sm mt-1">
              {activeTab === 'live' && liveSubtitle}
              {activeTab === 'metrics' && metricsSubtitle}
              {activeTab === 'retraining' && retrainingSubtitle}
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2 bg-zinc-900 border border-white/5 rounded-xl">
              <LockIcon className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-bold text-zinc-300 uppercase tracking-widest">
                {activeEncryptionProfile}
              </span>
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {activeTab === 'live' && (
            <motion.div
              key="live"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-6"
            >
              <div className="grid grid-cols-4 gap-6">
                {[
                  { label: 'Total Scanned', value: totalScanned, icon: Activity, color: 'text-blue-400' },
                  { label: 'Fraud Detected', value: fraudDetected, icon: AlertTriangle, color: 'text-red-400' },
                  { label: 'Quantum Keys', value: quantumKeys, icon: LockIcon, color: 'text-emerald-400' },
                  { label: 'Avg Risk', value: `${avgRisk}%`, icon: BarChart3, color: 'text-zinc-400' },
                ].map((stat, i) => (
                  <div key={i} className="p-6 bg-[#0A0A0A] border border-white/5 rounded-2xl">
                    <div className="flex items-center justify-between mb-4">
                      <stat.icon className={cn('w-5 h-5', stat.color)} />
                      <span className="text-[10px] font-bold text-zinc-600 uppercase tracking-widest">
                        Live
                      </span>
                    </div>
                    <p className="text-2xl font-bold tracking-tight">{stat.value}</p>
                    <p className="text-xs text-zinc-500 mt-1">{stat.label}</p>
                  </div>
                ))}
              </div>

              <div className="bg-[#0A0A0A] border border-white/5 rounded-2xl overflow-hidden">
                <div className="p-6 border-b border-white/5 flex items-center justify-between">
                  <h3 className="text-sm font-bold uppercase tracking-widest text-zinc-400">
                    Transaction Ledger
                  </h3>
                  <div className="flex items-center gap-2">
                    <div className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                      <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">
                        {filteredTransactions.length} Active Vectors
                      </span>
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-white/5 bg-zinc-900/30">
                        <th className="px-6 py-4 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">ID</th>
                        <th className="px-6 py-4 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Amount</th>
                        <th className="px-6 py-4 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Risk Score</th>
                        <th className="px-6 py-4 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Security</th>
                        <th className="px-6 py-4 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Label</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTransactions.map((t) => (
                        <tr
                          key={t.id}
                          className="border-b border-white/5 hover:bg-white/[0.02] transition-colors group"
                        >
                          <td className="px-6 py-4">
                            <span className="text-xs font-mono text-zinc-400">{t.id}</span>
                          </td>
                          <td className="px-6 py-4">
                            <span className="text-sm font-bold">${t.amount.toFixed(2)}</span>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-3">
                              <div className="flex-1 h-1.5 w-24 bg-zinc-800 rounded-full overflow-hidden">
                                <div
                                  className={cn(
                                    'h-full rounded-full transition-all duration-300',
                                    t.riskScore >= 75
                                      ? 'bg-red-500'
                                      : t.riskScore >= riskThreshold
                                      ? 'bg-yellow-500'
                                      : 'bg-emerald-500'
                                  )}
                                  style={{ width: `${t.riskScore}%` }}
                                />
                              </div>
                              <span
                                className={cn(
                                  'text-xs font-mono font-bold',
                                  t.riskScore >= riskThreshold
                                    ? 'text-red-400'
                                    : t.riskScore >= 25
                                    ? 'text-yellow-400'
                                    : 'text-emerald-400'
                                )}
                              >
                                {t.riskScore.toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <LockIcon className={cn('w-3 h-3', getSecurityColor(t.riskScore))} />
                              <span
                                className={cn(
                                  'text-[10px] font-bold uppercase tracking-widest',
                                  getSecurityColor(t.riskScore)
                                )}
                              >
                                {getSecurityVariant(t.riskScore)}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div
                              className={cn(
                                'inline-flex items-center gap-2 px-3 py-1 rounded-full border text-[10px] font-bold uppercase tracking-widest',
                                t.riskScore >= riskThreshold
                                  ? 'bg-red-500/10 border-red-500/20 text-red-400'
                                  : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                              )}
                            >
                              {t.riskScore >= riskThreshold ? (
                                <>
                                  <AlertTriangle className="w-3 h-3" />
                                  Fraud
                                </>
                              ) : (
                                <>
                                  <CheckCircle className="w-3 h-3" />
                                  Legitimate
                                </>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'metrics' && (
            <motion.div
              key="metrics"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid grid-cols-2 gap-8"
            >
              <div className="bg-[#0A0A0A] border border-white/5 rounded-2xl p-8">
                <h3 className="text-sm font-bold uppercase tracking-widest text-zinc-400 mb-8">
                  Confusion Matrix
                </h3>
                <div className="grid grid-cols-2 gap-4 aspect-square">
                  {[
                    {
                      label: 'True Negative',
                      value: metrics.confusionMatrix.tn,
                      sub: 'Legit correctly identified',
                      color: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                    },
                    {
                      label: 'False Positive',
                      value: metrics.confusionMatrix.fp,
                      sub: 'Legit flagged as fraud',
                      color: 'bg-red-500/10 border-red-500/20 text-red-400'
                    },
                    {
                      label: 'False Negative',
                      value: metrics.confusionMatrix.fn,
                      sub: 'Fraud missed by AI',
                      color: 'bg-orange-500/10 border-orange-500/20 text-orange-400'
                    },
                    {
                      label: 'True Positive',
                      value: metrics.confusionMatrix.tp,
                      sub: 'Fraud correctly identified',
                      color: 'bg-blue-500/10 border-blue-500/20 text-blue-400'
                    },
                  ].map((cell, i) => (
                    <div
                      key={i}
                      className={cn(
                        'p-6 rounded-2xl border flex flex-col justify-center items-center text-center',
                        cell.color
                      )}
                    >
                      <p className="text-3xl font-bold tracking-tight mb-1">{cell.value}</p>
                      <p className="text-[10px] font-bold uppercase tracking-widest opacity-80">
                        {cell.label}
                      </p>
                      <p className="text-[9px] mt-2 opacity-50">{cell.sub}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-8">
                <div className="bg-[#0A0A0A] border border-white/5 rounded-2xl p-8">
                  <h3 className="text-sm font-bold uppercase tracking-widest text-zinc-400 mb-8">
                    Model Performance
                  </h3>
                  <div className="space-y-6">
                    {[
                      { label: 'F1-Score', value: metrics.f1Score, color: 'bg-emerald-500' },
                      { label: 'Precision', value: metrics.precision, color: 'bg-blue-500' },
                      { label: 'Recall', value: metrics.recall, color: 'bg-purple-500' },
                    ].map((m, i) => (
                      <div key={i}>
                        <div className="flex justify-between mb-2">
                          <span className="text-xs font-bold text-zinc-500 uppercase tracking-widest">
                            {m.label}
                          </span>
                          <span className="text-xs font-mono text-white">
                            {(m.value * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-zinc-900 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${m.value * 100}%` }}
                            className={cn('h-full rounded-full', m.color)}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-[#0A0A0A] border border-white/5 rounded-2xl p-8">
                  <h3 className="text-sm font-bold uppercase tracking-widest text-zinc-400 mb-6">
                    Risk Distribution
                  </h3>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={transactions.slice(0, 10).map((t, i) => ({ name: i, risk: t.riskScore }))}
                      >
                        <defs>
                          <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <Area
                          type="monotone"
                          dataKey="risk"
                          stroke="#10b981"
                          fillOpacity={1}
                          fill="url(#colorRisk)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'retraining' && (
            <motion.div
              key="retraining"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-[#0A0A0A] border border-white/5 rounded-2xl p-8"
            >
              <div className="flex items-center gap-4 mb-10 p-6 bg-emerald-500/5 border border-emerald-500/10 rounded-2xl">
                <RefreshCw className="w-8 h-8 text-emerald-400 animate-spin-slow" />
                <div>
                  <h3 className="text-lg font-bold tracking-tight">Active Learning Loop</h3>
                  <p className="text-sm text-zinc-500">
                    The model is dynamically adjusting based on live transactions.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-8">
                <div className="col-span-2 space-y-4">
                  <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                    Recent High-Risk Transactions
                  </h4>

                  {feedbackTransactions.length === 0 ? (
                    <div className="p-10 text-center border border-dashed border-white/5 rounded-2xl">
                      <p className="text-sm text-zinc-600">No high-risk transactions yet.</p>
                    </div>
                  ) : (
                    feedbackTransactions.map((t) => (
                      <div
                        key={t.id}
                        className="p-4 bg-zinc-900/50 border border-white/5 rounded-xl flex items-center justify-between"
                      >
                        <div className="flex items-center gap-4">
                          <div
                            className={cn(
                              'w-2 h-2 rounded-full',
                              t.riskScore >= riskThreshold ? 'bg-red-500' : 'bg-emerald-500'
                            )}
                          />
                          <div>
                            <p className="text-xs font-mono text-zinc-300">{t.id}</p>
                            <p className="text-[10px] text-zinc-500 uppercase tracking-widest">
                              {t.riskScore >= riskThreshold ? 'Fraud' : 'Legitimate'}
                            </p>
                          </div>
                        </div>
                        <p className="text-xs font-mono text-emerald-400">
                          {t.riskScore.toFixed(1)}%
                        </p>
                      </div>
                    ))
                  )}
                </div>

                <div className="space-y-6">
                  <div className="p-6 bg-zinc-900/30 border border-white/5 rounded-2xl">
                    <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-4">
                      Retraining Progress
                    </h4>

                    <div className="flex items-center justify-center py-6">
                      <div className="relative w-32 h-32">
                        <svg className="w-full h-full transform -rotate-90">
                          <circle
                            cx="64"
                            cy="64"
                            r="60"
                            stroke="currentColor"
                            strokeWidth="8"
                            fill="transparent"
                            className="text-zinc-800"
                          />
                          <circle
                            cx="64"
                            cy="64"
                            r="60"
                            stroke="currentColor"
                            strokeWidth="8"
                            fill="transparent"
                            strokeDasharray={377}
                            strokeDashoffset={377 * (1 - retrainProgress / 100)}
                            className="text-emerald-500"
                          />
                        </svg>

                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className="text-2xl font-bold">{retrainProgress}%</span>
                          <span className="text-[8px] text-zinc-500 uppercase">Retrained</span>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={handleRetrain}
                      disabled={isRetraining}
                      className="w-full py-3 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed text-black text-xs font-bold uppercase tracking-widest rounded-xl transition-colors"
                    >
                      {isRetraining ? 'Retraining...' : 'Force Retrain Now'}
                    </button>

                    {lastRetrainTime && (
                      <p className="mt-3 text-[10px] text-zinc-500 text-center">
                        Last retrained at {lastRetrainTime}
                      </p>
                    )}

                    {retrainMessage && (
                      <p className="mt-2 text-[10px] text-center text-zinc-400">
                        {retrainMessage}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <style>{`
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin-slow {
          animation: spin-slow 8s linear infinite;
        }
      `}</style>
    </div>
  );
}
