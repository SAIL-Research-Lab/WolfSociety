# Prompt surfaces

Population agents use compact JSON prompts only:

- benign retail: risk appetite multiplier, skepticism, and optional public message
- pump-and-dump strategist: promotion intensity and dump timing
- finfluencer strategist: posting intensity and sell timing
- spoofing strategist: spoof size, cycles, and side bias
- wash-trading strategist: wash volume and withdrawal timing

Defense LLMs in E5 use the existing risk-only WolfGuard schema:

```json
{
  "asset_id": {
    "manipulation_risk": 0.0,
    "cascade_risk": 0.0,
    "confidence": 0.0
  }
}
```

The production backend is OpenRouter only. The mock backend is for local smoke
tests and never changes the production provider path.
