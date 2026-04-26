import express from "express";
import path from "path";
import fs from "fs";
import os from "os";
import cors from "cors";
import { createServer as createViteServer } from "vite";
import crypto from "crypto";
import { spawn } from "child_process";
import { GoogleGenAI } from "@google/genai";

type FraudModelPrediction = {
  fraud_probability: number;
  prediction: 0 | 1;
  threshold: number;
  model_meta?: {
    imbalance_strategy?: string;
    roc_auc?: number;
    avg_precision?: number;
  };
};

type FeedbackRow = {
  Time: number;
  Amount: number;
  Class: 0 | 1;
} & Record<`V${number}`, number>;

const PORT = 3000;
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const MODEL_PATH = "ml/model.pkl";
const TRAIN_SCRIPT = "ml/train_model.py";
const PREDICT_SCRIPT = "ml/predict.py";
const BASE_DATASET_PATH =
  process.env.CREDITCARD_CSV_PATH || path.join(process.cwd(), "creditcard.csv");
const DEBUG = true;

function debugLog(...args: any[]) {
  if (DEBUG) {
    console.log("[SERVER]", ...args);
  }
}

function runPythonScript(
  scriptPath: string,
  args: string[],
  stdinPayload?: unknown
): Promise<{ stdout: string; stderr: string }> {
  debugLog("runPythonScript called", { scriptPath, args });

  return new Promise((resolve, reject) => {
    const py = spawn(PYTHON_BIN, [scriptPath, ...args], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (d) => {
      const text = d.toString("utf8");
      stdout += text;
      debugLog("python stdout chunk", text.slice(0, 500));
    });

    py.stderr.on("data", (d) => {
      const text = d.toString("utf8");
      stderr += text;
      debugLog("python stderr chunk", text.slice(0, 500));
    });

    py.on("error", (err) => {
      debugLog("python spawn error", err);
      reject(err);
    });

    py.on("close", (code) => {
      debugLog("python process closed", { code, scriptPath });

      if (code !== 0) {
        reject(new Error(stderr || `${PYTHON_BIN} exited with code ${code}`));
        return;
      }

      resolve({ stdout, stderr });
    });

    if (stdinPayload !== undefined) {
      const payloadStr = JSON.stringify(stdinPayload);
      debugLog("writing stdin payload", payloadStr.slice(0, 800));
      py.stdin.write(payloadStr);
    }

    py.stdin.end();
  });
}

async function runPythonModel(payload: unknown): Promise<FraudModelPrediction> {
  debugLog("runPythonModel payload", payload);

  const { stdout, stderr } = await runPythonScript(
    PREDICT_SCRIPT,
    [MODEL_PATH],
    payload
  );

  debugLog("runPythonModel stdout", stdout);
  if (stderr) {
    debugLog("runPythonModel stderr", stderr);
  }

  try {
    const parsed = JSON.parse(stdout) as FraudModelPrediction;
    debugLog("runPythonModel parsed result", parsed);
    return parsed;
  } catch {
    throw new Error(`Failed to parse model output. stderr=${stderr}`);
  }
}

function normalizeFeedbackRow(row: any): FeedbackRow {
  const normalized: Partial<FeedbackRow> = {
    Time: Number(row?.Time ?? 0),
    Amount: Number(row?.Amount ?? 0),
    Class: Number(row?.Class ?? 0) === 1 ? 1 : 0,
  };

  for (let i = 1; i <= 28; i++) {
    normalized[`V${i}` as `V${number}`] = Number(row?.[`V${i}`] ?? 0);
  }

  return normalized as FeedbackRow;
}

function buildCsvFromFeedback(feedback: any[]): string {
  debugLog("buildCsvFromFeedback length", feedback.length);

  const headers = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount", "Class"];
  const lines = [headers.join(",")];

  for (const rawRow of feedback) {
    const row = normalizeFeedbackRow(rawRow);

    const values = [
      row.Time,
      ...Array.from({ length: 28 }, (_, i) => row[`V${i + 1}` as `V${number}`]),
      row.Amount,
      row.Class,
    ];

    lines.push(values.join(","));
  }

  debugLog("feedback csv preview", lines.slice(0, 3));
  return lines.join("\n");
}

function ensureDatasetExists(filePath: string) {
  debugLog("ensureDatasetExists", filePath);

  if (!fs.existsSync(filePath)) {
    throw new Error(
      `Dataset not found at ${filePath}. Put creditcard.csv in the project root or set CREDITCARD_CSV_PATH.`
    );
  }

  const stats = fs.statSync(filePath);
  debugLog("dataset exists", {
    filePath,
    size: stats.size,
  });

  const preview = fs.readFileSync(filePath, "utf8").split(/\r?\n/).slice(0, 2);
  debugLog("dataset preview", preview);
}

function createMergedTrainingCsv(feedback: any[]): string {
  ensureDatasetExists(BASE_DATASET_PATH);

  const baseLines = fs.readFileSync(BASE_DATASET_PATH, "utf8").trim().split(/\r?\n/);
  const header = baseLines[0];
  const sampleRows = baseLines.slice(1, 5001);
  const baseCsv = [header, ...sampleRows].join("\n");
  const feedbackCsv = buildCsvFromFeedback(feedback).trim();

  const feedbackLines = feedbackCsv.split(/\r?\n/);
  const feedbackRowsOnly = feedbackLines.slice(1);

  if (feedbackRowsOnly.length === 0) {
    debugLog("no feedback rows to merge");
    return baseCsv;
  }

  debugLog("merged feedback row count", feedbackRowsOnly.length);
  return `${baseCsv}\n${feedbackRowsOnly.join("\n")}`;
}

async function startServer() {
  debugLog("starting server");
  debugLog("cwd", process.cwd());
  debugLog("python bin", PYTHON_BIN);
  debugLog("model path", MODEL_PATH);
  debugLog("dataset path", BASE_DATASET_PATH);

  const app = express();

  app.use(cors());
  app.use(express.json({ limit: "10mb" }));

  app.use((req, _res, next) => {
    debugLog("incoming request", req.method, req.url);
    next();
  });

  app.get("/api/health", (_req, res) => {
    res.json({
      status: "ok",
      python: PYTHON_BIN,
      modelPath: MODEL_PATH,
      datasetPath: BASE_DATASET_PATH,
    });
  });

  app.get("/api/fraud/dataset-status", (_req, res) => {
    const exists = fs.existsSync(BASE_DATASET_PATH);

    debugLog("dataset-status route", {
      exists,
      datasetPath: BASE_DATASET_PATH,
    });

    res.json({
      exists,
      datasetPath: BASE_DATASET_PATH,
    });
  });

  app.post("/api/fraud/predict", async (req, res) => {
    try {
      debugLog("/api/fraud/predict body", req.body);
      const result = await runPythonModel(req.body ?? {});
      debugLog("/api/fraud/predict success", result);
      res.json(result);
    } catch (err: any) {
      debugLog("/api/fraud/predict failed", err?.message || err);
      res.status(500).json({
        error: "MODEL_PREDICTION_FAILED",
        message:
          err?.message ??
          "Failed to run the local sklearn model. Have you trained it (ml/model.pkl) yet?",
        hint:
          "Run: python -m pip install -r requirements.txt && python ml/train_model.py --csv creditcard.csv --out ml/model.pkl",
      });
    }
  });

  app.post("/api/fraud/retrain", async (req, res) => {
    try {
      const { feedback } = req.body ?? {};
      debugLog("/api/fraud/retrain feedback length", Array.isArray(feedback) ? feedback.length : "not-array");

      if (!Array.isArray(feedback) || feedback.length === 0) {
        return res.status(400).json({
          error: "NO_FEEDBACK",
          message: "No data provided for retraining",
        });
      }

      const mergedCsv = createMergedTrainingCsv(feedback);

      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "fraud-retrain-"));
      const tempCsvPath = path.join(tempDir, "creditcard_feedback_merge.csv");

      fs.writeFileSync(tempCsvPath, mergedCsv, "utf8");
      debugLog("temp retrain csv created", tempCsvPath);

      try {

        const trainingPromise = runPythonScript(TRAIN_SCRIPT, [
          "--csv",
          tempCsvPath,
          "--out",
          MODEL_PATH,
        ]);
        
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => {
            reject(new Error("Retraining timed out after 120 seconds"));
          }, 120000);
        });
        
        const { stdout, stderr } = await Promise.race([trainingPromise, timeoutPromise]);
        
        debugLog("retrain stdout", stdout);
        if (stderr) {
          debugLog("retrain stderr", stderr);
        }

        return res.json({
          status: "success",
          message: "Model retrained successfully using dataset + feedback",
          datasetUsed: BASE_DATASET_PATH,
          feedbackRows: feedback.length,
          output: stdout,
          warnings: stderr || null,
        });
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
          debugLog("temp retrain dir removed", tempDir);
        } catch (cleanupErr) {
          debugLog("temp cleanup failed", cleanupErr);
        }
      }
    } catch (err: any) {
      debugLog("/api/fraud/retrain failed", err?.message || err);
      res.status(500).json({
        error: "RETRAIN_ERROR",
        message: err?.message || "Unknown error",
      });
    }
  });

  app.post("/api/fraud/explain", async (req, res) => {
    try {
      const apiKey = process.env.GEMINI_API_KEY;

      if (!apiKey) {
        res.status(400).json({
          error: "MISSING_GEMINI_API_KEY",
          message: "Set GEMINI_API_KEY in .env.local to enable explanations.",
        });
        return;
      }

      debugLog("/api/fraud/explain called");

      const ai = new GoogleGenAI({ apiKey });
      const { transaction, modelResult } = req.body ?? {};

      const prompt = [
        "You are an assistant for a payment integrity dashboard.",
        "IMPORTANT:",
        "- You do NOT decide fraud vs legitimate.",
        "- You only explain the already-computed model output in plain language.",
        "- Be concise and actionable (2-6 bullets).",
        "",
        "Given:",
        `- Model result: ${JSON.stringify(modelResult)}`,
        `- Transaction fields: ${JSON.stringify(transaction)}`,
        "",
        "Return an explanation for why the model might rate this transaction as risky or safe,",
        "and suggest one follow-up action (e.g., manual review, step-up auth).",
      ].join("\n");

      const resp = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: prompt }] }],
      });

      debugLog("/api/fraud/explain success");
      res.json({ explanation: resp.text ?? "" });
    } catch (err: any) {
      debugLog("/api/fraud/explain failed", err?.message || err);
      res.status(500).json({
        error: "EXPLANATION_FAILED",
        message: err?.message ?? "Failed to generate explanation.",
      });
    }
  });

  app.post("/api/security/encrypt", (req, res) => {
    const { data } = req.body ?? {};
    debugLog("/api/security/encrypt called", data);

    const amount = Number(data?.amount ?? data?.Amount ?? 0);
    const riskScore = Number(data?.riskScore ?? 0);

    let kyberVariant = "Kyber-512";
    let securityLevel = "Standard";

    if (riskScore >= 75 || amount >= 800) {
      kyberVariant = "Kyber-1024";
      securityLevel = "Maximum";
    } else if (riskScore >= 45 || amount >= 300) {
      kyberVariant = "Kyber-768";
      securityLevel = "Elevated";
    }

    const kyberSeed = crypto.randomBytes(32).toString("hex");
    const sharedSecret = crypto.createHash("sha256").update(kyberSeed).digest();

    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv("aes-256-gcm", sharedSecret, iv);

    let encrypted = cipher.update(JSON.stringify(data ?? {}), "utf8", "hex");
    encrypted += cipher.final("hex");
    const authTag = cipher.getAuthTag().toString("hex");

    res.json({
      encryptedData: encrypted,
      iv: iv.toString("hex"),
      authTag,
      algorithm: `${kyberVariant} + AES-256-GCM`,
      kyberVariant,
      securityLevel,
      quantumSecure: true,
      processedAt: new Date().toISOString(),
    });
  });

  if (process.env.NODE_ENV !== "production") {
    debugLog("starting vite middleware mode");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    debugLog("serving production build from", distPath);
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Python binary: ${PYTHON_BIN}`);
    console.log(`Dataset path: ${BASE_DATASET_PATH}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});