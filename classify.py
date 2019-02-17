from fastai.vision import open_image
from fastai.vision.learner import load_learner
import sys

#defaults.device = torch.device('cpu')
if (len(sys.argv) != 2):
    print("USAGE: classify.py <path to image>")
    sys.exit(1)

img = open_image(sys.argv[1])
learn = load_learner(".", fname="its_raining_men.pkl")

pred_class,pred_idx,outputs = learn.predict(img)
print("p(ramen) = %1.2f, p(soba) = %1.2f, p(udon) = %1.2f. I think this is %s." % (outputs[0],outputs[1],outputs[2],pred_class))

