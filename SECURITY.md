# Security and privacy

QuietRecall is designed for local, user-owned Markdown. The HTTP service binds
to loopback only and authenticates requests with a local bearer token.

The repository intentionally excludes real memory corpora, generated indexes,
service tokens, transcripts, diagnostic logs, model weights, and machine-local
paths. Do not commit any of those files in a fork.

Memory content is exposed in two steps: a small title directory is issued for a
turn, then one item may be opened with an opaque, short-lived capability. Vault
scopes require explicit search authorization and are not opened by casual topic
mentions.

This is an experimental personal system, not a hardened multi-user secret
store. Anyone who can read the local process memory, token file, transcript, or
memory directory should be treated as trusted. Do not bind the service to a
public interface.

Please report vulnerabilities through a private GitHub security advisory rather
than a public issue.
