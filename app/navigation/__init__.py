"""
navigation — reads the true state from dynamics (read-only, never writes to it), adds sensor noise/delay/drift, outputs an estimated_state. That's its entire job — it's a one-way transform, not an update.


"""