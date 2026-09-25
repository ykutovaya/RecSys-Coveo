from hypothesis import settings

settings.register_profile(
    "normative", max_examples=1000, deadline=None, database=None, print_blob=True
)
settings.load_profile("normative")
