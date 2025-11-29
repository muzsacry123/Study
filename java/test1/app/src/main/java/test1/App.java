package test1.app.src.main.java.test1;

import java.util.Arrays;
import java.util.Date;
import java.awt.*;

public class App {
    public static void main(String[] args) {
        short   myAge   = -130;
        short   yourAge = myAge;
        long    abc     = 3123456789L;
        float   price   = 12.34F;
        char    letter  = 'B';
        boolean isGood  = true;
        
        // 时间 (java.util.Date)
        Date now = new Date();
            now.getTime();
        System.out.println(now);
        
        // 指针传递与值传递 (java.awt.*)
        int x = 1;  // primitive type
        int y = x;  // 此处为值传递
        Point point1 = new Point(1,1);  // reference type
        Point point2 = point1; // 此处为指针传递

        // 字符串
        String msg1 = new String("   Hello world!    ");
        String msg2 = "Hello world" + "!!";
            boolean isExclaim   = msg2.endsWith("!");
            int     strLen      = msg2.length();
            int     itemIdx     = msg2.indexOf("world");
            String  newString   = msg2.replace("!", "*");
            String  lowerCased  = msg2.toLowerCase();
            String  trimmed     = msg1.trim();
        System.out.println("Ends with \"!!\" : " + isExclaim);
        System.out.println("Length of string: " + strLen);
        System.out.println("World appears in: " + itemIdx);
        System.out.println("New String: " + newString);
        System.out.println("Lower Case: " + lowerCased);
        System.out.println("Trimmed: " + trimmed);

        // 数组 (import java.util.Arrays)
        int[] num = new int[5];
            num[0] = 1;
            num[1] = 2;
        String myArray = Arrays.toString(num);
        System.out.println("\nArray: " + myArray);
        
        int[] num2 = { 3, 6, 7, 4, 5 };
        Arrays.sort(num2);
        System.out.println("\nSorted array: " + Arrays.toString(num2));
        
    }
}
