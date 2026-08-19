"""The en dash is banned in source too, not only in prose formats.

The range below is written with an en dash – and that is the violation this
fixture pins. Both U+2014 and U+2013 are rejected; the passing fixtures cover
the flags that look similar and must not be.
"""

RANGE_NOTE = "opsets 17–23"
