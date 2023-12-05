import numpy as np

f = open("O3_cross_section_Serdyuchenko_2014.txt", "r")

lut = np.zeros(441)

count = 0
wavelength = 390
sum = 0.0
while True:
    
 
    line = f.readline()
    if not line:
        average = sum / float(count)
        lut[440] = average
        break
    
    linesplit = line.split()
    curr_wavelength = int(float(linesplit[0]))
    extinction = float(linesplit[1])

    if curr_wavelength != wavelength:
        average = sum / float(count)
        lut[wavelength-390] = average

        count = 0
        sum = 0.0
        wavelength = curr_wavelength
        

    sum += extinction
    count += 1
 
    

lut = lut.astype('float32')
lut.tofile('ozone_cross_section.dat')
print("finished generation")