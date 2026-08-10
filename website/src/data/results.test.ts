import { describe, expect, it } from 'vitest'
import { alphaCritical, exponent, fittedAlphaCritical, scalingPoints } from './results'

describe('canonical paper results', () => {
  it('preserves the six measured collapse midpoints', () => {
    expect(scalingPoints).toHaveLength(6)
    expect(alphaCritical(100)).toBeCloseTo(0.0471428571)
    expect(alphaCritical(2000)).toBeCloseTo(0.022)
    expect(scalingPoints.at(-1)?.criticalCount).toBe(44)
  })

  it('preserves the reported boundary exponent', () => {
    expect(exponent.estimate).toBeCloseTo(0.2224120555)
    expect(exponent.confidenceInterval[0]).toBeGreaterThan(0)
    expect(exponent.confidenceInterval[1]).toBeLessThan(1)
  })

  it('keeps the fitted boundary monotonic between measured sizes', () => {
    expect(fittedAlphaCritical(100)).toBeGreaterThan(fittedAlphaCritical(500))
    expect(fittedAlphaCritical(500)).toBeGreaterThan(fittedAlphaCritical(2000))
    expect(fittedAlphaCritical(750)).toBeGreaterThan(0)
  })
})
