package com.zavier.pratice;

/*
剑指offer练习题
 */


import java.util.*;

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

// 题目：调整数组顺序使奇数位于偶数前面
class Solution13 {
    public void reOrderArray(int[] array) {
        if (array.length == 0) return;

        List<Integer> odd = new ArrayList<>();
        List<Integer> even = new ArrayList<>();
        for (int ele :
                array) {
            if ((ele & 1) == 1) {
                odd.add(ele);
            } else {
                even.add(ele);
            }
        }
        int index = 0;
        for (Integer integer : odd) {
            array[index++] = integer;
        }
        for (Integer integer : even) {
            array[index++] = integer;
        }
    }
}

// 题目：链表中倒数第k个结点
class Solution14 {
    public ListNode FindKthToTail(ListNode head, int k) {
        ListNode fast, slow;
        int i = 0;
        fast = slow = head;
        for (; fast != null; i++) {
            if (i >= k) {
                slow = slow.next;
            }
            fast = fast.next;
        }
        return i < k ? null : slow;
    }
}

// 题目：反转链表
// 思路：递归
class Solution15 {
    public ListNode ReverseList(ListNode head) {
        if (head == null) return null;

        ListNode next = head.next;
        if (head.next == null) {
            return head;
        }
        ListNode result = ReverseList(head.next);
        next.next = head;
        head.next = null;
        return result;
    }
}

// 题目：反转链表
// 思路：迭代,主要的思想是用两个指针，其中newHead指向的是反转成功的链表的头部，currentHead指向的是还没有反转的链表的头部，
// 初始状态是newHead指向null，currentHead指向的是第一个元素，一直往后遍历直到newHead指向最后一个元素为止.
class Solution15_2 {
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode newHead = null;
        ListNode currentHead = head;
        while (currentHead != null) {
            ListNode next = currentHead.next;
            currentHead.next = newHead;
            newHead = currentHead;
            currentHead = next;
        }

        return newHead;
    }
}

// 题目：合并两个排序的链表
// 思路1：迭代，每次取当前两个链表节点中最小的加入新的链表中
class Solution16 {
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;

        ListNode result = new ListNode(-1);
        ListNode current = result;
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                current.next = list1;
                list1 = list1.next;
                current = current.next;
            } else {
                current.next = list2;
                list2 = list2.next;
                current = current.next;
            }
        }
        if (list1 != null) {
            current.next = list1;
        }
        if (list2 != null) {
            current.next = list2;
        }

        return result.next;
    }
}

// 题目：合并两个排序的链表
// 思路2：递归，同归并排序
class Solution16_1 {
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;

        ListNode result = null;
        if (list1.val <= list2.val) {
            result = list1;
            result.next = Merge(list1.next, list2);
        } else {
            result = list2;
            result.next = Merge(list1, list2.next);
        }

        return result;
    }
}

// 题目：树的子结构
// 思路：遍历树root1，如果节点与roo2相等，则判断是否有相同的子结构
class Solution17 {
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }
        if (root1.val == root2.val) {
            if (checkSubTree(root1, root2)) {
                return true;
            }
        }
        return HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

    boolean checkSubTree(TreeNode n1, TreeNode n2) {
        if (n2 == null) return true;
        if (n1 == null) return false;
        if (n1.val != n2.val) return false;

        return checkSubTree(n1.left, n2.left) && checkSubTree(n1.right, n2.right);
    }
}

// 题目：二叉树镜像
class Solution18 {
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode left = root.left;
        root.left = root.right;
        root.right = left;
        Mirror(root.left);
        Mirror(root.right);
    }
}

// 题目：顺时针打印矩阵
// 思路：逆时针旋转矩阵，每次输出第一行，然后去掉第一行
class Solution19 {
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        if (matrix.length == 0 || matrix[0].length == 0) {
            return result;
        }

        int[][] currentMatrix = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                currentMatrix[i][j] = matrix[i][j];
            }
        }

        for (; ; ) {
            for (int i = 0; i < currentMatrix[0].length; i++) {
                result.add(currentMatrix[0][i]);
            }
            if (result.size() == matrix.length * matrix[0].length) {
                break;
            }

            currentMatrix = rotateMatrix(currentMatrix, 1);
        }

        return result;
    }

    public int[][] rotateMatrix(int[][] matrix, int startIndex) {
        int[][] result = new int[matrix[startIndex].length][matrix.length - startIndex];
        for (int i = 0; i < matrix[startIndex].length; i++) {
            for (int j = 0; j < matrix.length - startIndex; j++) {
                result[i][j] = matrix[j + startIndex][matrix[startIndex].length - 1 - i];
            }
        }
        return result;
    }
}

//题目：包含min函数的栈
class Solution20 {
    Stack<Integer> s = new Stack<>();
    Stack<Integer> minStack = new Stack<>();

    public void push(int node) {
        s.push(node);
        if (minStack.empty() || node <= minStack.peek()) {
            minStack.push(node);
        }
    }

    public void pop() {
        if (s.empty() || minStack.empty()) return;

        if (minStack.peek().equals(s.peek())) {
            minStack.pop();
        }
        s.pop();
    }

    public int top() {
        return s.peek();
    }

    public int min() {
        return minStack.peek();
    }
}

// 题目：栈的压入、弹出序列
// 思路1：根据出栈队列，模拟入栈和出栈过程
class Solution21 {
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0) return false;

        Stack<Integer> s = new Stack<>();
        int pushIndex = 0;
        for (int i = 0; i < popA.length; i++) {
            while (s.empty() || s.peek() != popA[i]) {
                if (pushIndex >= pushA.length) {
                    break;
                }
                s.push(pushA[pushIndex++]);
            }
            if (s.peek() == popA[i]) {
                s.pop();
            } else if (pushIndex >= pushA.length) {
                break;
            }
        }
        return s.empty();
    }
}

// 题目：栈的压入、弹出序列
// 思路2：根据出栈队列，模拟出栈，要注意连续出栈的情况（while）
class Solution21_1 {
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0) return false;

        Stack<Integer> s = new Stack<>();
        int popIndex = 0;
        for (int i :
                pushA) {
            s.push(i);
            while (!s.empty() && s.peek() == popA[popIndex]) {
                s.pop();
                popIndex++;
                if (popIndex >= popA.length) break;
            }
        }
        return s.empty();
    }
}

// 题目：从上往下打印二叉树
class Solution22 {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if(root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode currentNode = queue.poll();
            result.add(currentNode.val);
            if(currentNode.left!=null) queue.offer(currentNode.left);
            if(currentNode.right!=null) queue.offer(currentNode.right);
        }
        return result;
    }
}

public class Main {
    public static void main(String[] args) {
        Solution21_1 s = new Solution21_1();
        int[] push = new int[]{1, 2, 3, 4, 5};
        int[] pop = new int[]{4, 5, 3, 2, 1};
        System.out.println(s.IsPopOrder(push, pop));
    }
}
