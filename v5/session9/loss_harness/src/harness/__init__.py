"""Session 9 loss-harness package: a small decoder-only transformer, a
separately-instantiable output head, and the loss/memory utilities the
assignment asks for.

Nothing here is exotic. The point of the exercise is the harness around
the model (shapes, shifting, masking, perplexity, memory), not the model
itself, so the architecture is kept deliberately small and readable.
"""
