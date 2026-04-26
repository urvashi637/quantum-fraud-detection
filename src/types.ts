export interface Transaction {
  id: string;
  time: number;
  amount: number;
  v1: number;
  v2: number;
  v3: number;
  v4: number;
  v5: number;
  v6: number;
  v7: number;
  v8: number;
  v9: number;
  v10: number;
  v11: number;
  v12: number;
  v13: number;
  v14: number;
  v15: number;
  v16: number;
  v17: number;
  v18: number;
  v19: number;
  v20: number;
  v21: number;
  v22: number;
  v23: number;
  v24: number;
  v25: number;
  v26: number;
  v27: number;
  v28: number;
  riskScore: number;
  isFraud: boolean;
  isQuantumSecure: boolean;
  status: 'pending' | 'verified' | 'flagged';
  userLabel?: 'legitimate' | 'fraud';
  explanation?: string;
}

export interface SecurityMetrics {
  f1Score: number;
  precision: number;
  recall: number;
  confusionMatrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}
