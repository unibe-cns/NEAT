
from pynestml.frontend.pynestml_frontend import generate_nest_compartmental_target


generate_nest_compartmental_target(
    input_path="tmp/multichannel_test/multichannel_test_model.nestml",
    target_path="tmp/multichannel_test/",
    module_name="tttt_module",
    logging_level="DEBUG"
)
