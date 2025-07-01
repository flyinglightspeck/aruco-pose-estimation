#include <SPI.h>
#include "LCD_Driver.h"
#include "GUI_Paint.h"
#include "image.h"

void setup()
{
  Config_Init();
  LCD_Init();
  LCD_Clear(WHITE);
  LCD_SetBacklight(50);
  Paint_NewImage(LCD_WIDTH, LCD_HEIGHT, 0, WHITE);
  Paint_Clear(WHITE);
  Paint_SetRotate(180);
	
  
  int code[] = {1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  int i, j;

/*********************************************************************************************************

To adjust the size of the aruco marker, change the value of "UWORD w" variable bellow based on the follwing table. 
E.g., setting w to 10 generates a 5.9 mm by 5.9 mm marker on the screean.

W         |  marker size (mm)
----------|--------
30        |  17.6
20        |  11.7
10        |  5.9
9	        |  5.3
8	        |  4.7
7	        |  4.1
6	        |  3.5
5	        |  2.9
4	        |  2.3
3	        |  1.8
2	        |  1.2
1	        |  0.6

*********************************************************************************************************/

  UWORD size = 6;
  UWORD w = 10;
  UWORD o = 120 - size/2 * w;

  for (j = 0; j < size; j++) {
    for (i = 0; i < size; i++) {
      if (i == 0 || j == 0 || i == size-1 || j == size-1) {
        Paint_DrawRectangle(o+i*w, o+j*w, o+i*w+w-1, o+j*w+w, BLACK, DOT_PIXEL_1X1, DRAW_FILL_FULL);
      } else if (code[(j-1)*(size-2)+i-1] == 0) {
        Paint_DrawRectangle(o+i*w, o+j*w, o+i*w+w-1, o+j*w+w, BLACK, DOT_PIXEL_1X1, DRAW_FILL_FULL);
      }
    }
  }

}
void loop()
{
  
}

