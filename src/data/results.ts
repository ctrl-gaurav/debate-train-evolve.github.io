export interface ModelResult {
  model: string
  modelFamily: string
  params: string
  gsm8k: { original: number; mad: number; evolved: number; delta: number }
  gsmPlus: { original: number; mad: number; evolved: number; delta: number }
  math: { original: number; mad: number; evolved: number; delta: number }
  arcC: { original: number; mad: number; evolved: number; delta: number }
  gpqa: { original: number; mad: number; evolved: number; delta: number }
}

export const mainResults: ModelResult[] = [
  {
    model: 'Qwen-2.5-1.5B',
    modelFamily: 'Qwen',
    params: '1.5B',
    gsm8k: { original: 62.77, mad: 72.33, evolved: 73.09, delta: 10.32 },
    gsmPlus: { original: 42.00, mad: 53.33, evolved: 55.92, delta: 13.92 },
    math: { original: 45.08, mad: 50.68, evolved: 52.20, delta: 7.12 },
    arcC: { original: 69.21, mad: 68.52, evolved: 68.36, delta: -0.85 },
    gpqa: { original: 19.42, mad: 18.75, evolved: 20.10, delta: 0.68 },
  },
  {
    model: 'Qwen-2.5-3B',
    modelFamily: 'Qwen',
    params: '3B',
    gsm8k: { original: 84.08, mad: 85.14, evolved: 86.05, delta: 1.97 },
    gsmPlus: { original: 61.75, mad: 68.00, evolved: 69.50, delta: 7.75 },
    math: { original: 61.36, mad: 65.72, evolved: 67.10, delta: 5.74 },
    arcC: { original: 83.53, mad: 84.64, evolved: 83.95, delta: 0.42 },
    gpqa: { original: 28.12, mad: 29.24, evolved: 30.50, delta: 2.38 },
  },
  {
    model: 'Qwen-2.5-7B',
    modelFamily: 'Qwen',
    params: '7B',
    gsm8k: { original: 90.67, mad: 91.21, evolved: 88.32, delta: -2.35 },
    gsmPlus: { original: 68.62, mad: 74.17, evolved: 74.71, delta: 6.09 },
    math: { original: 73.08, mad: 75.58, evolved: 77.20, delta: 4.12 },
    arcC: { original: 87.22, mad: 91.64, evolved: 90.89, delta: 3.67 },
    gpqa: { original: 32.81, mad: 33.71, evolved: 35.20, delta: 2.39 },
  },
  {
    model: 'Qwen-2.5-14B',
    modelFamily: 'Qwen',
    params: '14B',
    gsm8k: { original: 92.80, mad: 93.33, evolved: 93.74, delta: 0.94 },
    gsmPlus: { original: 71.79, mad: 77.25, evolved: 78.88, delta: 7.09 },
    math: { original: 76.18, mad: 78.62, evolved: 80.10, delta: 3.92 },
    arcC: { original: 90.27, mad: 93.77, evolved: 93.13, delta: 2.86 },
    gpqa: { original: 41.29, mad: 42.19, evolved: 43.60, delta: 2.31 },
  },
  {
    model: 'Llama-3.2-3B',
    modelFamily: 'Llama',
    params: '3B',
    gsm8k: { original: 72.55, mad: 73.84, evolved: 75.06, delta: 2.51 },
    gsmPlus: { original: 45.67, mad: 51.12, evolved: 53.79, delta: 8.12 },
    math: { original: 39.76, mad: 41.90, evolved: 43.80, delta: 4.04 },
    arcC: { original: 73.12, mad: 76.19, evolved: 77.23, delta: 4.11 },
    gpqa: { original: 26.12, mad: 29.24, evolved: 30.80, delta: 4.68 },
  },
  {
    model: 'Llama-3.1-8B',
    modelFamily: 'Llama',
    params: '8B',
    gsm8k: { original: 81.73, mad: 82.18, evolved: 86.81, delta: 5.08 },
    gsmPlus: { original: 55.62, mad: 60.79, evolved: 66.17, delta: 10.55 },
    math: { original: 46.66, mad: 47.90, evolved: 49.40, delta: 2.74 },
    arcC: { original: 77.65, mad: 85.07, evolved: 86.53, delta: 8.88 },
    gpqa: { original: 27.46, mad: 32.37, evolved: 34.10, delta: 6.64 },
  },
]

export interface CrossDomainResult {
  model: string
  fineTunedOn: string
  gsm8k: number | null
  gsmPlus: number | null
  arcEasy: number
  arcChallenge: number
  commonsenseQA: number
}

export const crossDomainResults: CrossDomainResult[] = [
  { model: 'Qwen-2.5-1.5B', fineTunedOn: 'GSM8K', gsm8k: null, gsmPlus: 51.21, arcEasy: 85.02, arcChallenge: 69.88, commonsenseQA: 64.29 },
  { model: 'Qwen-2.5-1.5B', fineTunedOn: 'GSM-Plus', gsm8k: 73.09, gsmPlus: null, arcEasy: 85.10, arcChallenge: 69.45, commonsenseQA: 64.21 },
  { model: 'Qwen-2.5-3B', fineTunedOn: 'GSM8K', gsm8k: null, gsmPlus: 65.54, arcEasy: 93.94, arcChallenge: 84.30, commonsenseQA: 75.92 },
  { model: 'Qwen-2.5-3B', fineTunedOn: 'GSM-Plus', gsm8k: 86.50, gsmPlus: null, arcEasy: 94.15, arcChallenge: 84.13, commonsenseQA: 75.92 },
  { model: 'Qwen-2.5-7B', fineTunedOn: 'GSM8K', gsm8k: null, gsmPlus: 69.63, arcEasy: 96.42, arcChallenge: 91.72, commonsenseQA: 82.96 },
  { model: 'Qwen-2.5-7B', fineTunedOn: 'GSM-Plus', gsm8k: 91.81, gsmPlus: null, arcEasy: 96.38, arcChallenge: 90.87, commonsenseQA: 82.88 },
  { model: 'Qwen-2.5-14B', fineTunedOn: 'GSM8K', gsm8k: null, gsmPlus: 73.46, arcEasy: 98.19, arcChallenge: 93.69, commonsenseQA: 83.70 },
  { model: 'Qwen-2.5-14B', fineTunedOn: 'GSM-Plus', gsm8k: 93.33, gsmPlus: null, arcEasy: 97.98, arcChallenge: 94.28, commonsenseQA: 82.23 },
  { model: 'Llama-3.2-3B', fineTunedOn: 'GSM8K', gsm8k: null, gsmPlus: 52.38, arcEasy: 87.12, arcChallenge: 72.01, commonsenseQA: 68.14 },
  { model: 'Llama-3.2-3B', fineTunedOn: 'GSM-Plus', gsm8k: 76.35, gsmPlus: null, arcEasy: 86.57, arcChallenge: 69.20, commonsenseQA: 68.55 },
  { model: 'Llama-3.1-8B', fineTunedOn: 'GSM8K', gsm8k: null, gsmPlus: 63.75, arcEasy: 93.01, arcChallenge: 84.39, commonsenseQA: 74.12 },
  { model: 'Llama-3.1-8B', fineTunedOn: 'GSM-Plus', gsm8k: 86.88, gsmPlus: null, arcEasy: 93.98, arcChallenge: 85.49, commonsenseQA: 73.87 },
]

export interface GRPOStepResult {
  model: string
  dataset: string
  baseTrain: number | null
  baseTest: number
  mad: number
  grpoTemp08: { s2k: number; s5k: number; s10k: number }
  grpoTemp02: { s2k: number; s5k: number; s10k: number }
}

export const grpoDetailedResults: GRPOStepResult[] = [
  { model: 'Qwen-2.5-1.5B', dataset: 'GSM8K', baseTrain: 81.55, baseTest: 62.77, mad: 72.33, grpoTemp08: { s2k: 67.78, s5k: 71.42, s10k: 71.04 }, grpoTemp02: { s2k: 73.09, s5k: 66.49, s10k: 53.98 } },
  { model: 'Qwen-2.5-3B', dataset: 'GSM8K', baseTrain: 91.28, baseTest: 84.08, mad: 85.14, grpoTemp08: { s2k: 85.06, s5k: 85.14, s10k: 86.13 }, grpoTemp02: { s2k: 84.00, s5k: 86.05, s10k: 84.38 } },
  { model: 'Qwen-2.5-7B', dataset: 'GSM8K', baseTrain: 94.29, baseTest: 90.67, mad: 91.21, grpoTemp08: { s2k: 88.32, s5k: 86.73, s10k: 84.00 }, grpoTemp02: { s2k: 86.96, s5k: 86.35, s10k: 88.02 } },
  { model: 'Qwen-2.5-14B', dataset: 'GSM8K', baseTrain: 94.89, baseTest: 92.80, mad: 93.33, grpoTemp08: { s2k: 87.72, s5k: 89.84, s10k: 91.81 }, grpoTemp02: { s2k: 86.58, s5k: 89.34, s10k: 93.74 } },
  { model: 'Llama-3.2-3B', dataset: 'GSM8K', baseTrain: 83.90, baseTest: 72.55, mad: 73.84, grpoTemp08: { s2k: 69.22, s5k: 21.53, s10k: 2.73 }, grpoTemp02: { s2k: 72.40, s5k: 75.06, s10k: 3.26 } },
  { model: 'Llama-3.1-8B', dataset: 'GSM8K', baseTrain: 89.08, baseTest: 81.73, mad: 82.18, grpoTemp08: { s2k: 84.61, s5k: 85.29, s10k: 85.22 }, grpoTemp02: { s2k: 86.81, s5k: 84.91, s10k: 0.15 } },
  { model: 'Qwen-2.5-1.5B', dataset: 'GSM-Plus', baseTrain: 42.40, baseTest: 42.00, mad: 51.62, grpoTemp08: { s2k: 47.49, s5k: 54.46, s10k: 19.00 }, grpoTemp02: { s2k: 52.33, s5k: 53.04, s10k: 55.92 } },
  { model: 'Qwen-2.5-3B', dataset: 'GSM-Plus', baseTrain: 61.14, baseTest: 61.75, mad: 67.79, grpoTemp08: { s2k: 66.21, s5k: 66.71, s10k: 69.13 }, grpoTemp02: { s2k: 64.04, s5k: 67.25, s10k: 68.25 } },
  { model: 'Qwen-2.5-7B', dataset: 'GSM-Plus', baseTrain: 68.27, baseTest: 68.62, mad: 74.17, grpoTemp08: { s2k: 64.71, s5k: 73.38, s10k: 74.71 }, grpoTemp02: { s2k: 67.75, s5k: 72.54, s10k: 74.50 } },
  { model: 'Qwen-2.5-14B', dataset: 'GSM-Plus', baseTrain: 71.11, baseTest: 71.79, mad: 77.25, grpoTemp08: { s2k: 70.79, s5k: 73.54, s10k: 75.88 }, grpoTemp02: { s2k: 73.00, s5k: 73.42, s10k: 75.62 } },
  { model: 'Llama-3.2-3B', dataset: 'GSM-Plus', baseTrain: 47.68, baseTest: 45.67, mad: 51.12, grpoTemp08: { s2k: 52.38, s5k: 53.29, s10k: 52.33 }, grpoTemp02: { s2k: 51.79, s5k: 49.54, s10k: 53.79 } },
  { model: 'Llama-3.1-8B', dataset: 'GSM-Plus', baseTrain: 58.56, baseTest: 55.62, mad: 60.79, grpoTemp08: { s2k: 64.96, s5k: 61.58, s10k: 66.17 }, grpoTemp02: { s2k: 65.08, s5k: 63.46, s10k: 60.46 } },
  { model: 'Qwen-2.5-1.5B', dataset: 'ARC-Challenge', baseTrain: null, baseTest: 69.21, mad: 68.52, grpoTemp08: { s2k: 30.03, s5k: 62.63, s10k: 68.36 }, grpoTemp02: { s2k: 47.27, s5k: 51.88, s10k: 67.51 } },
  { model: 'Qwen-2.5-3B', dataset: 'ARC-Challenge', baseTrain: null, baseTest: 83.53, mad: 84.64, grpoTemp08: { s2k: 81.66, s5k: 80.29, s10k: 83.63 }, grpoTemp02: { s2k: 81.91, s5k: 79.78, s10k: 83.95 } },
  { model: 'Qwen-2.5-7B', dataset: 'ARC-Challenge', baseTrain: null, baseTest: 87.22, mad: 91.64, grpoTemp08: { s2k: 88.57, s5k: 88.48, s10k: 90.63 }, grpoTemp02: { s2k: 88.43, s5k: 88.57, s10k: 90.89 } },
  { model: 'Qwen-2.5-14B', dataset: 'ARC-Challenge', baseTrain: null, baseTest: 90.27, mad: 93.77, grpoTemp08: { s2k: 91.81, s5k: 92.49, s10k: 93.13 }, grpoTemp02: { s2k: 91.47, s5k: 91.47, s10k: 92.67 } },
  { model: 'Llama-3.2-3B', dataset: 'ARC-Challenge', baseTrain: null, baseTest: 73.12, mad: 76.19, grpoTemp08: { s2k: 75.51, s5k: 74.32, s10k: 76.87 }, grpoTemp02: { s2k: 76.79, s5k: 74.57, s10k: 77.23 } },
  { model: 'Llama-3.1-8B', dataset: 'ARC-Challenge', baseTrain: null, baseTest: 77.65, mad: 85.07, grpoTemp08: { s2k: 83.70, s5k: 84.45, s10k: 86.03 }, grpoTemp02: { s2k: 84.98, s5k: 85.53, s10k: 86.53 } },
]

export const benchmarkNames: Record<string, string> = {
  gsm8k: 'GSM8K',
  gsmPlus: 'GSM-Plus',
  math: 'MATH',
  arcC: 'ARC-Challenge',
  gpqa: 'GPQA Main',
}

export const benchmarkInfo = [
  { name: 'GSM8K', type: 'Math', samples: '1,319 test', description: 'Grade school math word problems' },
  { name: 'GSM-Plus', type: 'Math', samples: '2,400 test', description: 'Adversarial math variations' },
  { name: 'MATH', type: 'Math', samples: '5,000 test', description: 'Competition-level mathematics' },
  { name: 'ARC-Easy', type: 'Science', samples: '2,376 test', description: 'Elementary science questions' },
  { name: 'ARC-Challenge', type: 'Science', samples: '1,172 test', description: 'Challenging science reasoning' },
  { name: 'GPQA Main', type: 'STEM', samples: '448 test', description: 'Graduate-level STEM questions' },
  { name: 'CommonsenseQA', type: 'Commonsense', samples: '1,140 test', description: 'Commonsense reasoning' },
]
