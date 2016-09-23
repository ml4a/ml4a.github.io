from PIL import Image

im = Image.open('/Users/gene/Teaching/ML4A/ml4a.github.io/images/test/tsne_caltech256_all.jpg')

w, h = im.width, im.height

for j in range(3):
	for i in range(3):
		x = float(i) * w / 3.
		y = float(j) * h / 3.
		print(x, y, w/3., h/3.)
		name = '/Users/gene/Teaching/ML4A/ml4a.github.io/images/test/tsne_caltech256_all_%d_%d.jpg'%(i, j)
		im_ = im.crop((x, y, w, h))
		im_.save(name, 'JPEG')
		
