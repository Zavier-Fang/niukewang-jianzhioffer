package com.zavier.pratice;

/*
剑指offer练习题
 */


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;

//题目：二维数组中的查找
//思路：从左下角开始找
class Solution1 {
    public boolean Find(int target, int[][] array) {
        int rows = array.length;
        if (rows == 0) return false;

        int cols = array[0].length;
        if (cols == 0) return false;

        for (int i = rows - 1; i >= 0; i--) {
            for (int j = 0; j < cols; j++) {
                if (target > array[i][j]) continue;
                else if (target < array[i][j]) break;
                else return true;
            }
        }
        return false;
    }
}

// 题目：替换空格
// 思路：从前往后数空格数量，然后从后往前遍历替换，可以减少移动次数
class Solution2 {
    public String replaceSpace(StringBuffer str) {
        int len = str.length();
        int spaceNum = 0;
        for (int i = 0; i < len; i++) {
            if (str.charAt(i) == ' ') {
                spaceNum++;
            }
        }
        str.setLength(len + spaceNum * 2);
        for (int i = len - 1; i >= 0; i--) {
            if (str.charAt(i) != ' ') {
                str.setCharAt(i + spaceNum * 2, str.charAt(i));
            } else {
                spaceNum--;
                str.setCharAt(i + spaceNum * 2, '%');
                str.setCharAt(i + spaceNum * 2 + 1, '2');
                str.setCharAt(i + spaceNum * 2 + 2, '0');
            }
        }
        return str.toString();
    }
}


class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}

// 题目：从尾到头打印链表
// 思路：递归
class Solution3 {
    ArrayList<Integer> result = new ArrayList<>();

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode != null) {
            printListFromTailToHead(listNode.next);
            result.add(listNode.val);
        }
        return result;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}

// 题目：重建二叉树
class Solution4 {
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        //数组长度为0的时候要处理
        if (pre.length == 0) return null;

        int rootVal = pre[0];

        //数组长度仅为1的时候就要处理
        if (pre.length == 1) return new TreeNode(rootVal);

        TreeNode root = new TreeNode(rootVal);

        //我们先找到root所在的位置，确定好前序和中序中左子树和右子树序列的范围
        int rootIndex = 0;
        for (int i = 0; i < in.length; i++) {
            if (in[i] == rootVal) {
                rootIndex = i;
                break;
            }
        }

        //递归，假设root的左右子树都已经构建完毕，那么只要将左右子树安到root左右即可
        //这里注意Arrays.copyOfRange(int[],start,end)是[)的区间
        root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, rootIndex + 1), Arrays.copyOfRange(in, 0, rootIndex));
        root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, rootIndex + 1, pre.length), Arrays.copyOfRange(in, rootIndex + 1, in.length));

        return root;
    }
}

// 题目：用两个栈实现队列
class Solution5 {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (stack2.empty()) {
            while (!stack1.empty()) {
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }
}

// 题目：旋转数组的最小数组
class Solution6 {
    public int minNumberInRotateArray(int[] array) {
        int len = array.length;
        if (len == 0) return 0;

        int result = array[0];
        for (int i = 0; i < len - 1; i++) {
            if (array[i + 1] < array[i]) {
                result = array[i + 1];
                break;
            }
        }
        return result;
    }
}

// 题目：斐波那契数列
class Solution7 {
    public int Fibonacci(int n) {
        int[] a = new int[40];
        a[0] = 0;
        a[1] = 1;
        for (int i = 2; i < 40; i++) {
            a[i] = a[i - 1] + a[i - 2];
        }
        return a[n];
    }
}

//题目：跳台阶
class Solution8 {
    public int JumpFloor(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;

        int[] f = new int[target + 1];
        f[1] = 1;
        f[2] = 2;
        for (int i = 3; i <= target; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[target];
    }
}

// 题目：变态跳台阶
class Solution9 {
    public int JumpFloorII(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;

        int[] f = new int[target + 1];
        f[1] = 1;
        for (int i = 2; i <= target; i++) {
            f[i] = 1;
            for (int j = 1; j < i; j++) {
                f[i] += f[j];
            }
        }
        return f[target];
    }
}

// 题目：变态跳台阶
/*
优化思路：
  因为：
    f(n) = f(n-1)+f(n-2)···+f(1)
    f(n-1) = f(n-1)+f(n-3)···+f(1)
  所以：
    f(n) = 2f(n-1)
 */
class Solution9_1 {
    public int JumpFloorII(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;

        int[] f = new int[target + 1];
        f[1] = 1;
        for (int i = 2; i <= target; i++) {
            f[i] = 2 * f[i - 1];
        }
        return f[target];
    }
}

// 题目：矩形覆盖
class Solution10 {
    public int RectCover(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;

        int[] f = new int[target + 1];
        f[1] = 1;
        f[2] = 2;
        for (int i = 3; i <= target; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[target];
    }
}

// 题目：二进制中1的个数
// 思路1：用1，不断左移和n的每位进行位与，来判断1的个数
class Solution11 {
    public int NumberOf1(int n) {
        int count = 0;
        int flag = 1;
        while (flag != 0) {
            if ((n & flag) != 0) {
                count++;
            }
            flag = flag << 1;
        }
        return count;
    }
}

// 题目：二进制中1的个数
// 思路2：如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0.
class Solution11_1 {
    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n = n & (n - 1);
        }
        return count;
    }
}

// 题目：数值的整数次方
// 思路：快速幂算法，
class Solution12 {
    public double Power(double base, int exponent) {
        if (base == 0) return 0;
        if (base == 1 || exponent == 0) return 1;

        int absExponent = Math.abs(exponent);
        double result = 1;
        while (absExponent > 0) {
//            if (absExponent % 2 == 0) {
//                //如果指数为偶数
//                absExponent /= 2; //把指数缩小为一半
//                base *= base; //底数变大成原来的平方
//            } else {
//                //如果指数为奇数
//                absExponent -= 1; //把指数减去1，使其变成一个偶数
//                result *= base; //此时记得要把指数为奇数时分离出来的底数的一次方收集好
//                absExponent /= 2; //此时指数为偶数，可以继续执行操作
//                base *= base;
//            }
            // 下面是优化，优化if-else条件，以及使用位运算
            if ((absExponent & 1) == 1) {
                absExponent -= 1;
                result *= base;
            }
            absExponent >>= 1;
            base *= base;
        }

        return exponent > 0 ? result : 1 / result;
    }
}

public class Main {
    public static void main(String[] args) {
        Solution12 s = new Solution12();
        System.out.println(s.Power(2, 5));
    }
}
