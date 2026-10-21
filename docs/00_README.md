# Micro Concepts for Valuation Models — Index

Spec pack for building valuation notebooks (`.ipynb`) in Claude Code.
Source lecture: **Class 1b — แนวคิดสำคัญในระดับจุลภาค (Micro Concepts)**, based on Mishkin Ch. 2, 4, 7.

## How to use this pack in Claude Code

1. Put this `docs/` folder in the repo root (`bond-debt/docs/`).
2. Tell Claude Code: *"Read `docs/00_README.md` and `docs/07_implementation_spec.md`, then build notebook NN following `docs/0X_*.md`."*
3. Each concept file has the same structure:
   - **Intuition** — what the concept means in plain language
   - **Formula** — the exact equation, with symbols defined
   - **Implementation notes** — function signature, edge cases, numerical method
   - **Test cases** — worked examples with known answers (all re-computed and verified)
4. `07_implementation_spec.md` is the build contract: package layout, conventions, notebook list, and the full test table.

## Files

| File | Covers | Slide pages |
|---|---|---|
| `01_financial_system.md` | Direct vs indirect finance, securities, bonds vs stocks, what an interest rate is, rate build-up, Thai rate chart | 2–5, 11–12 |
| `02_time_value_of_money.md` | Compounding, PV/FV, multi-cash-flow PV, compounding frequency, EAR, continuous compounding, annuity, perpetuity, growing perpetuity, cash-flow additivity, forward rates | 6–10 |
| `03_debt_valuation.md` | Four credit instruments, yield to maturity (YTM), current yield, discount bond yield, price–yield relationship, return vs interest rate, interest-rate risk, reinvestment risk, real vs nominal rate, duration and convexity, credit spreads | 4, 6–10 (+ Mishkin Ch. 4) |
| `04_equity_valuation.md` | One-period model, generalized dividend model, Gordon growth model, two-stage DDM, implied return and growth, P/E link, how the market sets prices | 13–17 |
| `05_risk_return_capm.md` | Return measures, risk, systematic vs unsystematic risk, diversification, CAPM, beta, equity risk premium, SML, alpha | 21–29 |
| `06_efficient_markets.md` | Rational expectations, EMH condition, arbitrage, random walk, weak/semi-strong/strong forms, empirical tests, behavioral finance | 18–20, 29 |
| `07_implementation_spec.md` | Package layout, coding conventions, notebook roadmap, data sources, master test table | — |

## Concept map

```mermaid
flowchart TD
    FS[Financial system<br/>direct & indirect finance] --> DM[Debt market<br/>bonds, loans]
    FS --> EM[Equity market<br/>common stock]
    TVM[Time value of money<br/>PV = CF / (1+i)^t] --> DM
    TVM --> EM
    DM --> YTM[Yield to maturity<br/>= IRR of a bond]
    YTM --> RET[Return vs interest rate<br/>R = i_c + g]
    RET --> IRR[Interest-rate risk<br/>duration, convexity]
    YTM --> REAL[Real vs nominal<br/>Fisher equation]
    EM --> DDM[Dividend models<br/>one-period → general → Gordon]
    CAPM[CAPM<br/>k_e = r_f + β(r_m − r_f)] -->|discount rate k_e| DDM
    RISK[Systematic vs<br/>unsystematic risk] --> CAPM
    DDM --> PRICE[Market price<br/>set by lowest-risk bidder]
    EMH[Efficient market hypothesis] --> PRICE
    RE[Rational expectations] --> EMH
```

The one rule that ties everything together (slide 13):

> **Value of any investment = present value of its expected future cash flows.**

Bonds, loans, and stocks only differ in (a) what the cash flows look like and (b) which discount rate you use. CAPM supplies the equity discount rate; EMH says the market price already reflects all available information about those cash flows and risks.

## Notation used across all files

| Symbol | Meaning (Thai) | Unit |
|---|---|---|
| $PV$ | present value (มูลค่าปัจจุบัน) | currency |
| $FV$, $CF_t$ | future value / cash flow at time $t$ (มูลค่าอนาคต / กระแสเงินสด) | currency |
| $i$ | interest rate / yield per year (อัตราดอกเบี้ย) | decimal (0.05 = 5%) |
| $t$, $n$ | time in years, number of periods | years / periods |
| $m$ | compounding (or coupon) periods per year — the existing notebook calls this `freq` | int |
| $C$ | coupon payment per year | currency |
| $F$ | face / par value | currency |
| $P$, $P_0$ | price today | currency |
| $D_t$ | dividend at end of year $t$ | currency |
| $k_e$ | required return on equity | decimal |
| $g$ | growth rate (dividends) or rate of capital gain (bonds) — context decides | decimal |
| $r_f$, $r_m$, $\beta$ | risk-free rate, expected market return, beta | decimal |
| $\pi^e$ | expected inflation | decimal |

## Sources and how they were used

- **Main:** `Class1b_MicroConcepts…pdf` (30 slides). Every concept on the slides is covered.
- **Mishkin, *Economics of Money, Banking, and Financial Markets* 12e (Global)** — Ch. 2, 4, 7: used for derivations, worked examples, and the Table 1 / Table 2 test values. The Ch. 4 duration appendix is online-only and not in the PDF, so the duration section uses standard textbook formulas.
- **CFA Level I 2026, Quantitative Methods** — Learning Modules 1 (Rates and Returns) and 2 (Time Value of Money in Finance): used for compounding frequency, annualizing, log returns, two-stage DDM, implied return/growth, P/E, cash-flow additivity, forward rates, and extra test cases.
- Items marked **[extension]** go beyond the slides. They are included because a valuation notebook needs them (for example, duration and beta estimation).
