from utils import *



class MergedHeatMap(object):
    """
    Merge cancer heatmap and lymp heatmap together.
    Input image arrays should be opencv type(BGR).
    """
    def __init__(self, cancerImg, lympImg):
        self.cancerImg = cancerImg
        self.lympImg = lympImg
        self.thresholding_cancer = 255*0.5
        self.thresholding_lym = 255*0.5
        self.mergedHeatMap = self.merge()

    def merge(self):
        cancerImg, lympImg = self.cancerImg, self.lympImg
        if np.max(cancerImg) < 2:
            cancerImg = (cancerImg*255).astype(np.uint8)

        if np.max(lympImg) < 2:
            lympImg = (lympImg*255).astype(np.uint8)

        filter = 3
        cancerImg = cv2.GaussianBlur(cancerImg[:, :, 2], (filter, filter), 0)
        lympImg = cv2.GaussianBlur(lympImg, (filter, filter), 0)

        up = int(math.ceil(lympImg.shape[0]/cancerImg.shape[0]))
        if up > 1:
            iml_u = np.zeros((cancerImg.shape[0] * up, cancerImg.shape[1] * up), dtype=np.float32)
            for x in range(cancerImg.shape[1]):
                for y in range(cancerImg.shape[0]):
                    iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = cancerImg[y, x]

            cancerImg = iml_u.astype(np.uint8)

        cancerImg = cv2.resize(cancerImg, (lympImg.shape[1], lympImg.shape[0]), interpolation=cv2.INTER_LINEAR)

        out = np.zeros_like(lympImg, dtype=np.uint8)
        for i in range(lympImg.shape[0]):
            for j in range(lympImg.shape[1]):
                b, g, r = lympImg[i, j]
                out[i, j] = np.array([192,192,192])
                is_tumor = (cancerImg[i, j] > self.thresholding_cancer)
                is_lym = (r > self.thresholding_lym)
                is_tissue = (b > 100)
                # Tissue, Tumor, Lym
                if not is_tissue:
                    out[i, j] = np.array([255,255,255]) #White
                    continue

                if is_tumor and (not is_lym):
                    out[i, j] = np.array([0,255,255]) #Yellow
                elif (not is_tumor) and is_lym:
                    out[i, j] = np.array([0,0,200]) #Redish
                elif is_tumor and is_lym:
                    out[i, j] = np.array([0,0,255]) #Red

        return out
