# Security Policy

## Supported versions

AutoDistiller is pre-1.0. Only the latest release gets security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a vulnerability

Please do not open a public issue.

Report privately through GitHub:
[Security > Report a vulnerability](https://github.com/OriAlpha/autodistiller/security/advisories/new).

Expect an acknowledgement within a few days. If a fix is needed, we will agree a
disclosure timeline with you and credit you in the advisory unless you would
rather stay anonymous.

## Scope

AutoDistiller loads models and datasets that you point it at, and runs them
locally. Two things are worth knowing:

- **`trust_remote_code` executes code from the model repository.** It is off by
  default and should stay off for any model you have not vetted. This is
  upstream Transformers behavior, not a flaw in AutoDistiller, but enabling it
  means running arbitrary code from that repo.
- **Run records embed your environment.** `record.json` contains hostname,
  hardware and library versions. Check before publishing one in an issue.

Reports about either are still welcome, especially if AutoDistiller makes an
unsafe path easier to hit than it should be.
