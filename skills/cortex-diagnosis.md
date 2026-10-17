# Skill: diagnose Cortex LLMHoster connectivity

Diagnose why MAgent cannot use Cortex without modifying Cortex or exposing credentials.

1. Read effective MAgent backend URL/model/auth configuration with secrets redacted.
2. Check root `/health`, then `/ready`, then `/v1/models`; preserve HTTP status and Cortex error body.
3. If auth fails, confirm whether a key is configured, but never print it.
4. Compare the requested/default model id with `/v1/models` and `/admin/api/runtime` state.
5. Check model capability compatibility and whether the default/local runtime is running.
6. Inspect Cortex metrics/runtime/hardware and bounded recent logs through documented admin routes.
7. Offer the smallest reversible action: correct URL/key/model, start one model, or restart only the failed runtime.

Never vendor, patch, or alter Cortex source. Do not stop all models or install updates without explicit approval. End with the failing layer, evidence, recommended command, and rollback.
