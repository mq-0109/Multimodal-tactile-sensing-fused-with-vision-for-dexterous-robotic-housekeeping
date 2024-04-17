
def decodedata(recdata):

    frame_right = False
    hotfilm_values = [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0]
    coldfilm_values= [0.0, 0.0, 0.0, 0.0]
    recdata = str(recdata)

    recdata = recdata.strip("b").strip("'")

    try:
        frame_start_pos_1 = recdata.index('A')

        frame_right = True
        if (frame_start_pos_1>len(recdata)-3 or frame_start_pos_1<3):
            if(frame_start_pos_1>len(recdata)-3):
                frame_start_pos_1=frame_start_pos_1-len(recdata)
            if(frame_start_pos_1==2):
                recdata = recdata[frame_start_pos_1+2:]
            else:
                recdata = recdata[frame_start_pos_1+2:frame_start_pos_1 - 2]
        else:
            cut1 = recdata[frame_start_pos_1 + 2:]
            cut2 = recdata[0: frame_start_pos_1 - 2]
            recdata = cut1 + cut2
        #print(recdata)

        hotfilm_values[0] = float(recdata[0:8])
        hotfilm_values[1] = float(recdata[18:26])
        hotfilm_values[2] = float(recdata[36:44])
        hotfilm_values[3] = float(recdata[54:62])
        hotfilm_values[4] = float(recdata[72:80])
        hotfilm_values[5] = float(recdata[90:98])
        hotfilm_values[6] = float(recdata[108:116])
        hotfilm_values[7] = float(recdata[126:134])

        coldfilm_values[0]= float(recdata[81:89])
        coldfilm_values[1]= float(recdata[99:107])
        coldfilm_values[2]= float(recdata[117:125])
        coldfilm_values[3]= float(recdata[135:143])


    except Exception as e:
        frame_right = False

    return frame_right, hotfilm_values,coldfilm_values

