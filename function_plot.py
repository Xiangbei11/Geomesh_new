from stl import mesh
from matplotlib import pyplot

# Load the STL files and add the vectors to the plot
conventional = mesh.Mesh.from_file('CAD_new/Boeing_777-9x_9236_notwatertight.stl')

pyplot.figure()
pyplot.title('xy')
num= 1
pyplot.scatter(conventional.x, conventional.y)
pyplot.grid(b=True, which='major', color='#000000') 
pyplot.minorticks_on()
pyplot.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=20)

x_l_list = [30.]
x_t_list = [48.]
y_l_list = [4]
y_t_list = [34.]
for i in range(1):
    pyplot.plot([x_l_list[i],x_t_list[i]],[y_l_list[i],y_t_list[i]], color='r')
pyplot.show()
