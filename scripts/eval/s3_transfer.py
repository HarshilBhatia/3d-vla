"""Minimal S3 download/upload for the online-eval container.

The eval image ships neither the AWS CLI nor boto3, so eval jobs install boto3 at
startup and use this helper instead of `aws s3 cp` / `aws s3 sync`.

Usage:
    python scripts/eval/s3_transfer.py down s3://bucket/key            /local/file
    python scripts/eval/s3_transfer.py sync-down s3://bucket/prefix/   /local/dir
    python scripts/eval/s3_transfer.py up   /local/file                s3://bucket/key
    python scripts/eval/s3_transfer.py sync-up   /local/dir            s3://bucket/prefix
"""

import os
import sys

import boto3


def _split(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    return bucket, key


def down(s3, uri, dest):
    bucket, key = _split(uri)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    s3.download_file(bucket, key, dest)
    print(f"downloaded {uri} -> {dest} ({os.path.getsize(dest)} bytes)", flush=True)


def sync_down(s3, uri, dest_dir):
    bucket, prefix = _split(uri)
    prefix = prefix.rstrip("/") + "/"
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            dest = os.path.join(dest_dir, key[len(prefix) :])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
            n += 1
    # A silent zero-object sync would surface later as a confusing "no
    # variations found" from RLBench, so fail loudly here instead.
    if n == 0:
        raise RuntimeError(f"no objects found under {uri}")
    print(f"synced {n} objects {uri} -> {dest_dir}", flush=True)


def up(s3, src, uri):
    bucket, key = _split(uri)
    s3.upload_file(src, bucket, key)
    print(f"uploaded {src} -> {uri}", flush=True)


def sync_up(s3, src_dir, uri):
    """Upload a directory tree, skipping the JSON results uploaded separately."""
    bucket, prefix = _split(uri)
    prefix = prefix.rstrip("/")
    n = 0
    for root, _, files in os.walk(src_dir):
        for name in files:
            if name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src_dir)
            s3.upload_file(path, bucket, f"{prefix}/{rel}")
            n += 1
    print(f"uploaded {n} files {src_dir} -> {uri}", flush=True)


def main():
    mode, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    s3 = boto3.client("s3")
    if mode == "down":
        down(s3, a, b)
    elif mode == "sync-down":
        sync_down(s3, a, b)
    elif mode == "up":
        up(s3, a, b)
    elif mode == "sync-up":
        sync_up(s3, a, b)
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
