import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/home/jara/PycharmProjects/GPUcluster/AdditionalFiles/RelVol_All_Dice_MainLobes3_Dice_loss.csv', sep=',')

dice = df['dice']
x = df['rel_vol']

plt.figure()
plt.scatter(x, dice, marker='.')
plt.xlabel('Relative volume to lobe volume (%)')
plt.ylabel('Binary Dice Coefficient')
plt.show()

df2 = pd.read_csv('/home/jara/PycharmProjects/GPUcluster/AdditionalFiles/RelVol_Per_Lobe_Dice_MainLobes3_Dice_loss.csv', sep=',')

x_TL = df2['rel_vol_TL']
dice_TL = df2['dice_TL']
x_TR = df2['rel_vol_TR']
dice_TR = df2['dice_TR']
x_FL = df2['rel_vol_FL']
dice_FL = df2['dice_FL']
x_FR = df2['rel_vol_FR']
dice_FR = df2['dice_FR']
x_CL = df2['rel_vol_CL']
dice_CL = df2['dice_CL']
x_CR = df2['rel_vol_CR']
dice_CR = df2['dice_CR']
x_OP = df2['rel_vol_OP']
dice_OP = df2['dice_OP']
x_AP = df2['rel_vol_AP']
dice_AP = df2['dice_AP']

plt.figure()
plt.scatter(x_TL, dice_TL, marker='.', linewidths=3, alpha=0.5)
plt.scatter(x_TR, dice_TR, marker='.', linewidths=3, alpha=0.5)
plt.scatter(x_CL, dice_CL, marker='.', linewidths=3, alpha=0.5)
plt.scatter(x_CR, dice_CR, marker='.', linewidths=3, alpha=0.5)
plt.scatter(x_FL, dice_FL, marker='.', linewidths=3, alpha=0.5)
plt.scatter(x_FR, dice_FR, marker='.', linewidths=3, alpha=0.5)
plt.scatter(x_OP, dice_OP, marker='.', linewidths=3, alpha=0.5)
plt.scatter(x_AP, dice_AP, marker='.', linewidths=3, alpha=0.5)
plt.xlabel('Relative volume (%)')
plt.ylabel('Binary Dice Coefficient')
plt.title('Relative volume to total lobe volume (excluding background)')
plt.legend({'Temporal Left', 'Temporal Right', 'Central Left', 'Central Right',
            'Frontal Left', 'Frontal Right', 'Occipital Parietal', 'Appendices'})
plt.show()

################################################################
# WDL_BG
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df3 = pd.read_csv('/home/jara/PycharmProjects/GPUcluster/AdditionalFiles/RelVol_Per_Lobe_Dice_MainLobes3_WDL_BG.csv', sep=' ')

x_FL = df3['rel_vol_FL']
dice_FL = df3['dice_FL']
x_FR = df3['rel_vol_FR']
dice_FR = df3['dice_FR']
x_CR = df3['rel_vol_CR']
dice_CR = df3['dice_CR']
x_AP = df3['rel_vol_AP']
dice_AP = df3['dice_AP']

plt.figure()
plt.scatter(x_CR, dice_CR, marker='.', linewidths=3, alpha=1)
plt.scatter(x_FL, dice_FL, marker='.', linewidths=3, alpha=1)
plt.scatter(x_FR, dice_FR, marker='.', linewidths=3, alpha=1)
plt.scatter(x_AP, dice_AP, marker='.', linewidths=3, alpha=1)
plt.xlabel('Relative volume (%)')
plt.ylabel('Binary Dice Coefficient')
plt.ylim((0, 1))
plt.title('Relative volume to total lobe volume (excluding background)')
plt.legend({'Central Right', 'Frontal Left', 'Frontal Right', 'Appendices'}, loc= 'lower right')
plt.show()

##########################################################################

# WDL_FG
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df3 = pd.read_csv('/home/jara/PycharmProjects/GPUcluster/AdditionalFiles/RelVol_Per_Lobe_Dice_MainLobes3_WDL_FG.csv', sep=',')

x_FL = df3['rel_vol_FL']
dice_FL = df3['dice_FL']
x_FR = df3['rel_vol_FR']
dice_FR = df3['dice_FR']
x_CR = df3['rel_vol_CR']
dice_CR = df3['dice_CR']
x_AP = df3['rel_vol_AP']
dice_AP = df3['dice_AP']

plt.figure()
plt.scatter(x_CR, dice_CR, marker='.', linewidths=3, alpha=1)
plt.scatter(x_FL, dice_FL, marker='.', linewidths=3, alpha=1)
plt.scatter(x_FR, dice_FR, marker='.', linewidths=3, alpha=1)
plt.scatter(x_AP, dice_AP, marker='.', linewidths=3, alpha=1)
plt.xlabel('Relative volume (%)')
plt.ylabel('Binary Dice Coefficient')
plt.ylim((0, 1))
plt.title('Relative volume to total lobe volume (excluding background)')
plt.legend({'Central Right', 'Frontal Left', 'Frontal Right', 'Appendices'}, loc= 'lower right')
plt.show()