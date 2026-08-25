# Local Logic Fix

Project-authored synthetic JavaScript fixture for the Phase 2 scorer foundation. The visible tests cover neighboring behavior; the controller-owned scorer checks the exact boundary named in the goal. Only `fixture/` is copied into the writable attempt root. The scorer and expected control overlays are never shown to or mounted inside that root.