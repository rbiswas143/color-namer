from log_utils import log

import color.data.colors_big as colors_big
import color.data.colors_small as colors_small

log.debug('test')

log.info(colors_big.load_color_names())
log.info(colors_small.load_color_names())
