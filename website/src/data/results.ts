export type ScalingPoint = {
  n: number
  alphaC: number
  criticalCount: number
  width1090?: number
}

export const scalingPoints: ScalingPoint[] = [
  { n: 100, alphaC: 0.047142857142857146, criticalCount: 4.714285714285714, width1090: 0.023714285714285716 },
  { n: 200, alphaC: 0.036, criticalCount: 7.2, width1090: 0.0128 },
  { n: 300, alphaC: 0.03666666666666667, criticalCount: 11, width1090: 0.028 },
  { n: 500, alphaC: 0.034444444444444444, criticalCount: 17.22222222222222, width1090: 0.017777777777777778 },
  { n: 1000, alphaC: 0.03, criticalCount: 30 },
  { n: 2000, alphaC: 0.022, criticalCount: 44 },
]

export const exponent = {
  estimate: 0.22241205549592397,
  confidenceInterval: [0.20637176283626152, 0.27651839195861005] as const,
}

export const interventions = {
  jointDecrease: 0.046,
  jointIncrease: -0.018,
  increasedReach: -0.017,
}

const fittedAmplitude = Math.exp(
  scalingPoints.reduce(
    (sum, point) => sum + Math.log(point.alphaC) + exponent.estimate * Math.log(point.n),
    0,
  ) / scalingPoints.length,
)

export function fittedAlphaCritical(n: number) {
  return fittedAmplitude * n ** -exponent.estimate
}

export function alphaCritical(n: number) {
  const measured = scalingPoints.find((point) => point.n === Math.round(n))
  return measured?.alphaC ?? fittedAlphaCritical(n)
}

export const provenance = {
  scaling: 'paper_experiments_v3/figures/generated/table2_scaling_results.csv',
  exponent: 'paper_experiments_v3/figures/generated/table3_scaling_exponent_bootstrap.csv',
  label: 'S1 primary analysis · 12 paired seeds · six resolved society sizes',
}
