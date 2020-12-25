from PIL import Image


def detile(tiles, size, top, left):
    """

    :param tiles: ndarray shape like (N, H, W)
    :param size: target size like (W, H)
    :param top:
    :param left:
    :return: PIL Image
    """
    im = Image.new('L', size)

    _, top_step, left_step = tiles.shape
    width, height = size

    i = 0
    for y in range(-top, -top + height, top_step):
        for x in range(-left, -left + width, left_step):
            tile = Image.fromarray(tiles[i])
            im.paste(tile, (x, y))
            i += 1

    return im
