#!/usr/bin/env python3
"""
Vector Store Index Update Script

Usage:
    python update_index.py --incremental  # Daily update (fast)
    python update_index.py --full         # Full rebuild (slow, monthly)
    python update_index.py --status       # Check index status
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.vectorstore import (force_rebuild_vectorstore, get_index_status,
                                  update_vectorstore_incremental)


def print_status():
    """Print current index status."""
    print("=" * 80)
    print("VECTORSTORE INDEX STATUS")
    print("=" * 80)

    status = get_index_status()

    print(f"\nIndex Exists: {'✅ Yes' if status['index_exists'] else '❌ No'}")
    print(f"Last Update: {status.get('last_update', 'Never')}")
    print(f"Total Records Indexed: {status.get('total_records', 0):,}")
    print(f"Total Chunks: {status.get('total_chunks', 0):,}")
    print(f"Version: {status.get('version', 'unknown')}")

    if status.get("pending_changes"):
        changes = status["pending_changes"]
        print(
            f"\n{'⚠️ ' if status.get('update_recommended') else '✅ '}Pending Changes:"
        )
        print(f"  • New Records: {changes['new_records']}")
        print(f"  • Deleted Records: {changes['deleted_records']}")

        if status.get("update_recommended"):
            print(f"\n💡 Recommendation: Run 'python update_index.py --incremental'")
        else:
            print(f"\n✅ Index is up to date!")

    print("=" * 80)


def run_incremental_update():
    """Run incremental update."""
    print("=" * 80)
    print("INCREMENTAL INDEX UPDATE")
    print("=" * 80)
    print("\n⏱️  This will only process new/changed data (typically takes 2-5 minutes)")
    print("Starting incremental update...\n")

    success = update_vectorstore_incremental()

    if success:
        print("\n" + "=" * 80)
        print("✅ INCREMENTAL UPDATE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nYour vector index is now up to date with the latest data.")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ INCREMENTAL UPDATE FAILED")
        print("=" * 80)
        print("\nTry running a full rebuild: python update_index.py --full")
        return 1


def run_full_rebuild():
    """Run full rebuild."""
    print("=" * 80)
    print("FULL INDEX REBUILD")
    print("=" * 80)
    print("\n⚠️  WARNING: This will take 3-4 hours due to rate limiting!")
    print("Use this option for:")
    print("  • Monthly maintenance")
    print("  • After major data changes")
    print("  • If incremental updates fail")

    response = input("\nContinue with full rebuild? (yes/no): ").strip().lower()

    if response != "yes":
        print("Cancelled.")
        return 1

    print("\nStarting full rebuild... ⏳")
    print("💡 Tip: This is a good time for a coffee break!\n")

    success = force_rebuild_vectorstore()

    if success:
        print("\n" + "=" * 80)
        print("✅ FULL REBUILD COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ FULL REBUILD FAILED")
        print("=" * 80)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Update ChromaDB vector store index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current status
  python update_index.py --status
  
  # Daily incremental update (recommended)
  python update_index.py --incremental
  
  # Full rebuild (monthly maintenance)
  python update_index.py --full
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Show index status and pending changes",
    )
    group.add_argument(
        "--incremental",
        "-i",
        action="store_true",
        help="Update index incrementally (fast, only new/changed data)",
    )
    group.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Full rebuild of index (slow, ~3-4 hours)",
    )

    args = parser.parse_args()

    try:
        if args.status:
            print_status()
            return 0
        elif args.incremental:
            return run_incremental_update()
        elif args.full:
            return run_full_rebuild()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
