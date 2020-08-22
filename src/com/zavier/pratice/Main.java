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
        if(target==0) return 0;
        if(target==1) return 1;

        int[] f = new int[target+1];
        f[1] = 1;
        for(int i=2;i<=target;i++){
            f[i] = 1;
            for (int j=1;j<i;j++){
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
        if(target==0) return 0;
        if(target==1) return 1;

        int[] f = new int[target+1];
        f[1] = 1;
        for(int i=2;i<=target;i++){
            f[i] = 2*f[i-1];
        }
        return f[target];
    }
}

// 题目：矩形覆盖
class Solution10 {
    public int RectCover(int target) {
        if (target==0) return 0;
        if (target==1) return 1;
        if (target==2) return 2;

        int[] f = new int[target+1];
        f[1] = 1;
        f[2] = 2;
        for (int i=3;i<=target;i++) {
            f[i] = f[i-1] + f[i-2];
        }
        return f[target];
    }
}

public class Main {
    public static void main(String[] args) {

    }
}
