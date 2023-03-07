

def findBolt(results):
    bolts = results.xyxy[0]
    count = bolts.shape[0]
    first = bolts[0]
    x = first[0]
    y = first[1]
    midx = first[2] + first[0]
    midy = first[3] + first[1]

    x_location = midx/2
    y_location = midy/2
    x_location = x_location.item()
    x_location = round(x_location)

    y_location = y_location.item()
    y_location = round(y_location)
    #print(x_location)
    coordinate = (x_location,y_location)
    return coordinate