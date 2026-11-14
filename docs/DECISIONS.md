# Architecture & Engineering Decisions Log (docs/DECISIONS.md)

| Date | Task ID | Decision | Rationale | Alternatives Considered |
| :--- | :--- | :--- | :--- | :--- |
| 2026-11-14 | T001 | Separate runtime vs pipeline dependencies | Streamlit Community Cloud deployment requires fast cold-start (<10s) and minimal install footprint without heavy geospatial compilers. | Single monolithic `requirements.txt` (rejected due to bloat and deployment timeouts). |
| 2026-11-14 | T001 | Pre-fetch and commit offline demo site data | Rule 4 strictly forbids runtime dependence on live NASA APIs during judging evaluation. Offline Parquet ensures 100% demo reliability. | Pure live API calls on every session (rejected due to network latency/outage risk). |
| 2026-11-14 | T001 | Treat Random Forest as an Emulator / Surrogate | Prevents circular ML penalty under Validity criterion. The RF model acts as a fast sensitivity surrogate, not a live yield predictor. | Claiming ML predicts crop yield (rejected as scientifically invalid without ground-truth harvest records). |
