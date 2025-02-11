# flar/__init__.py

# Let users import the two versions easily:
from .flar_small import flar as run_flar_small
from .flar_large import flar as run_flar_large

# Now the user can do:
#    from flar import flar_small, flar_large
# or
#    import flar
#    flar.run_flar_small(...)
