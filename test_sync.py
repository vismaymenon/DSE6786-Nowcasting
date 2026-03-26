"""
test_sync.py
============
Temporary test script. Delete after confirming sync works.
Run from project root: python test_sync.py
"""

from database.client import get_backend_client
from pipeline.fred_loader import sync_csv_to_supabase

client = get_backend_client()
sync_csv_to_supabase(client)
'''

---

**Before you run it, one thing to check:**

Make sure your `pipeline/` and `supabase/` folders each have an `__init__.py` file. Without it, Python won't treat them as packages and the imports will fail.

Create empty files at:
- `pipeline/__init__.py`
- `supabase/__init__.py`

They can be completely empty — their presence is all that matters.

Once those are in place, run:
'''