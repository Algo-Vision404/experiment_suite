# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability.

Report the affected component, reproduction steps, impact, and relevant logs or proof of concept through a private security channel available to the repository maintainers.

EXSS processes datasets and can persist derived artifacts. Do not commit secrets, credentials, private datasets, or sensitive production records.

## Scope

Security reports are especially useful for unsafe deserialization or artifact loading, path traversal in artifact handling, leakage of sensitive dataset contents, dependency vulnerabilities, and command execution through untrusted configuration or input.
