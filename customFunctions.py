import sqlite3

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

#Call this function to create database and table
def startDatabase():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("CREATE TABLE xarm (x REAL, y REAL, z REAL)")
    c.execute("CREATE TABLE xarm1 (x REAL, y REAL)")
    #commit changes
    print("database created")
    conn.commit()
    #close connection
    conn.close()
    return

def fetchdataReal():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("SELECT * FROM xarm")
    x = 0
    y = 0
    z = 0
    items = c.fetchall()
    if len(items) != 0:
        for item in items:
            x = x + item[0]
            y = y + item[1]
            z = z + item[2]
        x = x/len(items)
        x = round(x, 1)
        y = y/len(items)
        y = round(y, 1)
        z = z/len(items)
        z = round(z, 1)
    return x, y, z

def fetchdataCamera():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("SELECT * FROM xarm1")
    x = 0
    y = 0
    items = c.fetchall()
    if len(items) != 0:
        for item in items:
            x = x + item[0]
            y = y + item[1]
        x = x/len(items)
        x = round(x)
        y = y/len(items)
        y = round(y)
    return x, y

def stopDatabase():
    conn = sqlite3.connect('xarm.db')
    c = conn.cursor()
    c.execute("DROP TABLE xarm")
    c.execute("DROP TABLE xarm1")
    conn.commit()
    print("database deleted")
    #close connection
    conn.close()
    return
#stopDatabase()
#fetchdata()
#startDatabase()
