
def load_dataset(imageFile, maskFile):
    images = np.load(imageFile)
    labels = np.load(maskFile)

    images = np.float32(images)
    labels = np.float32(labels)
    images = (images - 127.5) / 127.5
    old = labels
    new = np.empty((len(labels),256, 256,3))
    new.fill(-1)
    new[:,:,:,:3] = old
    labels = np.float32(new)

    return [images, labels]