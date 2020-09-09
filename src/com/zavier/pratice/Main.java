package com.zavier.pratice;

/*
剑指offer练习题
 */


import sun.awt.image.ImageWatched;

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
        if (root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode currentNode = queue.poll();
            result.add(currentNode.val);
            if (currentNode.left != null) queue.offer(currentNode.left);
            if (currentNode.right != null) queue.offer(currentNode.right);
        }
        return result;
    }
}

// 题目：二叉搜索树的后序遍历
// 思路：二叉搜索树特点：根节点大于左子树的所有节点，小于左子树的所有节点，后序遍历最后一个元素为根节点
class Solution23 {
    public boolean VerifySquenceOfBST(int[] sequence) {
        int size = sequence.length;
        if (size == 0) {
            return false;
        }
        if (size == 1) {
            return true;
        }

        int mid = sequence[size - 1];
        int left = -1, right = size - 1;
        for (int i = 0; i < size; i++) {
            if (sequence[i] < mid) {
                left = i;
            } else {
                break;
            }
        }
        for (int i = size - 2; i >= 0; i--) {
            if (sequence[i] > mid) {
                right = i;
            } else {
                break;
            }
        }
        if (left >= right || Math.abs((right - left)) > 1) {
            return false;
        }
        boolean result = true;
        if (left != -1) {
            result &= VerifySquenceOfBST(Arrays.copyOfRange(sequence, 0, left + 1));
            if (!result) {
                return false;
            }
        }

        if (right != size - 1) {
            result &= VerifySquenceOfBST(Arrays.copyOfRange(sequence, right, size - 1));
        }
        return result;
    }
}

// 题目：二叉树中和为某一值的路径
class Solution24 {
    ArrayList<ArrayList<Integer>> result = new ArrayList<>();
    ArrayList<Integer> resultItem = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        findPathRecursive(root, target);
        return result;
    }

    void findPathRecursive(TreeNode root, int target) {
        if (root == null || target - root.val < 0) {
            return;
        }

        resultItem.add(root.val);
        if (root.left == null && root.right == null && target - root.val == 0) {
            result.add(new ArrayList<>(resultItem));
        }

        if (target - root.val > 0) {
            findPathRecursive(root.left, target - root.val);
            findPathRecursive(root.right, target - root.val);
        }
        resultItem.remove(resultItem.size() - 1);
    }
}

class RandomListNode {
    int label;
    RandomListNode next = null;
    RandomListNode random = null;

    RandomListNode(int label) {
        this.label = label;
    }
}

// 题目：复杂链表的复制
// 思路：利用哈希表保存每个clone的副本
class Solution25 {
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) return null;
        HashMap<RandomListNode, RandomListNode> map = new HashMap<>();
        RandomListNode current = pHead;
        while (current != null) {
            map.put(current, new RandomListNode(current.label));
            current = current.next;
        }
        current = pHead;
        while (current != null) {
            map.get(current).next = map.get(current.next);
            map.get(current).random = map.get(current.random);
            current = current.next;
        }
        return map.get(pHead);
    }

}

// 题目：二叉搜索树与双向链表
/*
1 思路：明确Convert函数的功能。
  输入：输入一个二叉搜索树的根节点。
  过程：将其转化为一个有序的双向链表。
  输出：返回该链表的头节点。

2 明确成员变量pLast的功能。
 pLast用于记录当前链表的末尾节点。

3 明确递归过程。
  递归的过程就相当于按照中序遍历，将整个树分解成了无数的小树，
  然后将他们分别转化成了一小段一小段的双向链表。再利用pLast记录总的链表的末尾，然后将这些小段链表一个接一个地加到末尾。
 */
class Solution26 {
    TreeNode pLast = null;

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) return null;
        TreeNode head = Convert(pRootOfTree.left);

        if (head == null) {
            head = pRootOfTree;
        }

        pRootOfTree.left = pLast;
        if (pLast != null) {
            pLast.right = pRootOfTree;
        }
        pLast = pRootOfTree;

        Convert(pRootOfTree.right);

        return head;
    }
}

// 题目：字符串的排列
// 思路：递归法，问题转换为先固定第一个字符，求剩余字符的排列；求剩余字符排列时跟原问题一样。
//
//(1) 遍历出所有可能出现在第一个位置的字符（即：依次将第一个字符同后面所有字符交换）；
//
//(2) 固定第一个字符，求后面字符的排列（即：在第1步的遍历过程中，插入递归进行实现）。
class Solution27 {
    private ArrayList<String> result = new ArrayList<>();

    public ArrayList<String> Permutation(String str) {
        if (str == null || str.length() == 0) {
            return result;
        }

        findPermutation(str.toCharArray(), 0);
        Collections.sort(result);

        return result;
    }

    void findPermutation(char[] str, int begin) {
        if (begin == str.length - 1) {
            String S = String.valueOf(str);
            if (!result.contains(S)) {
                result.add(S);
            }
            return;
        }
        for (int i = begin; i < str.length; i++) {
            swap(str, begin, i);
            findPermutation(str, begin + 1);
            swap(str, begin, i);
        }
    }

    void swap(char[] str, int i, int j) {
        char temp = str[i];
        str[i] = str[j];
        str[j] = temp;
    }
}

// 题目：数组中出现次数超过一半的数字
class Solution28 {
    public int MoreThanHalfNum_Solution(int[] array) {
        Map<Integer, Integer> m = new HashMap<>();
        for (int ele : array) {
            int count = m.getOrDefault(ele, 0);
            m.put(ele, count + 1);
            if (count + 1 > array.length / 2) {
                return ele;
            }
        }
        return 0;
    }
}

// 题目：最小的k个数
// 思路：优先队列解决（最大堆）
class Solution29 {
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<>();
        if (input.length < k || k == 0) {
            return result;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>(k, (Integer o1, Integer o2) -> o2 - o1);
        for (int i = 0; i < k; i++) {
            queue.add(input[i]);
        }
        for (int i = k; i < input.length; i++) {
            if (input[i] < queue.peek()) {
                queue.add(input[i]);
                queue.poll();
            }
        }
        while (!queue.isEmpty()) {
            result.add(queue.poll());
        }

        return result;
    }
}

// 题目：连续子数组最大和
// 思路：假设f(n)为以n结尾的最大的子数组的和，则有f(n) = max(f(n-1)+array[n], array[n])
class Solution30 {
    public int FindGreatestSumOfSubArray(int[] array) {
        int[] f = new int[array.length];
        if (array.length == 1) {
            return array[0];
        }

        f[0] = array[0];
        int ret = f[0];
        for (int i = 1; i < array.length; i++) {
            f[i] = Math.max(f[i - 1] + array[i], array[i]);
            ret = Math.max(ret, f[i]);
        }
        return ret;
    }
}

// 题目：把数组排成最小的数
// 思路：
//   先将整型数组转换成String数组，然后将String数组排序，最后将排好序的字符串数组拼接出来。关键就是制定排序规则。
//   排序规则如下：
//   若ab > ba 则 a > b，
//   若ab < ba 则 a < b，
//   若ab = ba 则 a = b；
//   解释说明：
//   比如 "3" < "31"但是 "331" > "313"，所以要将二者拼接起来进行比较
class Solution32 {
    public String PrintMinNumber(int[] numbers) {
        String[] s = new String[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            s[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(s, (String s1, String s2) -> (s1 + s2).compareTo(s2 + s1));
        StringBuilder sb = new StringBuilder();
        for (String str :
                s) {
            sb.append(str);
        }
        return sb.toString();
    }
}

// 题目：丑数
// 思路：要注意，后面的丑数是有前一个丑数乘以2，3，5中的一个得来。因此可以用动态规划去解
// 同时注意一下，题目意思应该是质数因此，而不是因子，因为8的因子有1，2，4，8
class Solution33 {
    public int GetUglyNumber_Solution(int index) {
        if (index < 1) return 0;
        int[] result = new int[index];
        result[0] = 1;

        int t1 = 0, t2 = 0, t3 = 0;
        for (int i = 1; i < index; i++) {
            result[i] = Math.min(Math.min(result[t1] * 2, result[t2] * 3), result[t3] * 5);
            if (result[i] == result[t1] * 2) t1++;
            if (result[i] == result[t2] * 3) t2++;
            if (result[i] == result[t3] * 5) t3++;
        }

        return result[index - 1];
    }
}

// 题目：第一次只出现一次的字符
class Solution34 {
    public int FirstNotRepeatingChar(String str) {
        Map<Character, Integer> map = new LinkedHashMap<>();
        for (int i = 0; i < str.length(); i++) {
            char s = str.charAt(i);
            int number = map.getOrDefault(s, 0);
            map.put(s, number + 1);
        }

        Character result = null;
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getValue() == 1) {
                result = entry.getKey();
                break;
            }
        }
        if (result == null) return -1;
        else {
            return str.indexOf(result);
        }
    }
}

// 题目：数组中的逆序对
// 思路：分治的思想，归并排序的应用
class Solution35 {
    int cnt = 0;

    public int InversePairs(int[] array) {

        if (array.length != 0) {
            divide(array, 0, array.length - 1);
        }
        return cnt;
    }

    // 分
    void divide(int[] array, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = (start + end) / 2;

        divide(array, start, mid);
        divide(array, mid + 1, end);

        merge(array, start, mid, end);
    }

    // 治
    void merge(int[] array, int start, int mid, int end) {
        int[] temp = new int[end - start + 1];

        int i = start, j = mid + 1, k = 0;
        //下面就开始两两进行比较，若前面的数大于后面的数，就构成逆序对
        while (i <= mid && j <= end) {
            if (array[i] < array[j]) {
                temp[k++] = array[i++];
            } else {
                temp[k++] = array[j++];
                //a[i]>a[j]了，那么这一次，从a[i]开始到a[mid]必定都是大于这个a[j]的，因为此时分治的两边已经是各自有序了
                cnt = (cnt + mid - i + 1) % 1000000007;
            }
        }
        while (i <= mid) temp[k++] = array[i++];
        while (j <= end) temp[k++] = array[j++];

        for (int l = 0; l < k; l++) {
            array[l + start] = temp[l];
        }
    }
}

// 题目：两个链表的第一个公共节点
/*思路：
        假定 List1长度: a+n  List2 长度:b+n, 且 a<b
        那么 p1 会先到链表尾部, 这时p2 走到 a+n位置,将p1换成List2头部
        接着p2 再走b+n-(n+a) =b-a 步到链表尾部,这时p1也走到List2的b-a位置，还差a步就到可能的第一个公共节点。
        将p2 换成 List1头部，p2走a步也到可能的第一个公共节点。如果恰好p1==p2,那么p1就是第一个公共节点。  或者p1和p2一起走n步到达列表尾部，二者没有公共节点，退出循环。 同理a>=b.
        时间复杂度O(n+a+b)

        其实就是两个链表相加（a+n+b+n),然后走a+b+n步即可。
*/
class Solution36 {
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode p1 = pHead1, p2 = pHead2;

        while (p1 != p2) {
            if (p1 != null) p1 = p1.next;
            if (p2 != null) p2 = p2.next;
            if (p1 != p2) {
                if (p1 == null) p1 = pHead2;
                if (p2 == null) p2 = pHead1;
            }
        }

        return p1;
    }
}

// 题目：数字在升序数组中出现的次数
// 思路：二分查找，找到最后出现的k的下标和最先出现的k的下标
class Solution37 {
    public int GetNumberOfK(int[] array, int k) {
        return getLastIndex(array, k) - getFirstIndex(array, k) + 1;
    }

    //获取k第一次出现的下标
    int getFirstIndex(int[] array, int k) {
        int start = 0, end = array.length - 1;

        int mid = (start + end) / 2;

        while (start <= end) {
            if (array[mid] < k) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
            mid = (start + end) / 2;
        }

        return start;
    }

    //获取k最后出现的下标
    int getLastIndex(int[] array, int k) {
        int start = 0, end = array.length - 1;

        int mid = (start + end) / 2;

        while (start <= end) {
            if (array[mid] <= k) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
            mid = (start + end) / 2;
        }

        return end;
    }
}

// 题目：二叉树的深度
class Solution38 {
    public int TreeDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(TreeDepth(root.left), TreeDepth(root.right)) + 1;
    }
}

// 题目：平衡二叉树
// 思路1：分别求每个节点的的树的高度，只有满足每个节点的子树高度差小于等于1才是平衡二叉树
class Solution39 {
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) return true;

        int leftDepth = treeDepth(root.left);
        int rithtDepth = treeDepth(root.right);

        if (Math.abs(leftDepth - rithtDepth) > 1) return false;

        return IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
    }

    int treeDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(treeDepth(root.left), treeDepth(root.right)) + 1;
    }
}

// 题目：平衡二叉树
// 思路2:修改求树高度的函数，当左右子树高度差大于1时停止递归
class Solution39_1 {
    public boolean IsBalanced_Solution(TreeNode root) {
        return treeDepth(root) >= 0;
    }

    int treeDepth(TreeNode root) {
        if (root == null) return 0;
        int left = treeDepth(root.left);
        if (left == -1) return -1;
        int right = treeDepth(root.right);
        if (right == -1) return -1;

        return Math.abs(left - right) <= 1 ? Math.max(left, right) + 1 : -1;
    }
}

// 题目：数组中只出现一次的数字
/* 思路：
首先：位运算中异或的性质：两个相同数字异或=0，一个数和0异或还是它本身。

当只有一个数出现一次时，我们把数组中所有的数，依次异或运算，最后剩下的就是落单的数，因为成对儿出现的都抵消了。

依照这个思路，我们来看两个数（我们假设是AB）出现一次的数组。我们首先还是先异或，剩下的数字肯定是A、B异或的结果，
这个结果的二进制中的1，表现的是A和B的不同的位。我们就取第一个1所在的位数，假设是第3位，接着把原数组分成两组，
分组标准是第3位是否为1。如此，相同的数肯定在一个组，因为相同数字所有位都相同，而不同的数，肯定不在一组。
然后把这两个组按照最开始的思路，依次异或，剩余的两个结果就是这两个只出现一次的数字。
 */
class Solution40 {
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        int xorResult = 0;
        if (array.length == 2) {
            num1[0] = array[0];
            num2[0] = array[1];
            return;
        }

        for (int i = 0; i < array.length; i++) {
            xorResult ^= array[i];
        }
        int index = findFirst1Index(xorResult);
        for (int i = 0; i < array.length; i++) {
            if ((array[i] >> index & 1) == 0) {
                num1[0] ^= array[i];
            } else {
                num2[0] ^= array[i];
            }
        }
    }

    int findFirst1Index(int a) {
        int index = 0;
        while (a != 0) {
            if ((a >> index & 1) == 1) {
                return index;
            }
            index++;
        }
        return index;
    }
}

// 题目：和为s的连续正序列
class Solution41 {
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        //两个起点，相当于动态窗口的两边，根据其窗口内的值的和来确定窗口的位置和大小
        int pleft = 1, pright = 2;
        while (pleft < pright && pright < sum) {
            //由于是连续的，差为1的一个序列，那么求和公式是(a0+an)*n/2
            int curSum = (pleft + pright) * (pright - pleft + 1) / 2;
            // 值相等，放进结果集，然后左边界往右移动
            if (curSum == sum) {
                ArrayList<Integer> list = new ArrayList<>();
                for (int i = pleft; i <= pright; i++) list.add(i);
                result.add(list);
                //如果当前窗口内的值之和小于sum，那么右边窗口右移一下
            } else if (curSum < sum) {
                pright++;
                //如果当前窗口内的值之和大于sum，那么左边窗口右移一下
            } else {
                pleft++;
            }
        }
        return result;
    }
}

// 题目：和为s的两个数字
/* 思路：
    假设：若b>a,且存在，
    a + b = s;
    (a - m ) + (b + m) = s
    则：(a - m )(b + m)=ab - (b-a)m - m*m < ab；说明外层的乘积更小
    也就是说依然是左右夹逼法！！！只需要2个指针
    1.left开头，right指向结尾
    2.如果和小于sum，说明太小了，left右移寻找更大的数
    3.如果和大于sum，说明太大了，right左移寻找更小的数
    4.和相等，把left和right的数返回
 */
class Solution42 {
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> result = new ArrayList<>();
        int left = 0, right = array.length - 1;
        while (left <= right) {
            if (array[left] + array[right] == sum) {
                result.add(array[left]);
                result.add(array[right]);
                break;
            } else if (array[left] + array[right] > sum) {
                right--;
            } else {
                left++;
            }
        }
        return result;
    }
}

// 题目：左循环字符串
class Solution43 {
    public String LeftRotateString(String str, int n) {
        int length = str.length();
        char[] chars = new char[length];
        for (int i = 0; i < length; i++) {
            int newIndex = (i - n + length) % length;
            chars[newIndex] = str.charAt(i);
        }
        return String.valueOf(chars);
    }
}

// 题目：翻转单词顺序列
class Solution44 {
    public String ReverseSentence(String str) {
        String[] s = str.split(" ");
        if (s.length == 0) return str;

        StringBuilder sb = new StringBuilder();
        for (int i = s.length - 1; i >= 0; i--) {
            sb.append(s[i]);
            if (i != 0) {
                sb.append(" ");
            }
        }
        return sb.toString();
    }
}

// 题目：扑克牌顺子
class Solution45 {
    public boolean isContinuous(int[] numbers) {
        if (numbers.length == 0) return false;
        if (numbers.length == 1) return true;

        Arrays.sort(numbers);
        int cntOf0 = 0;
        for (int i = 0; i < numbers.length - 1; i++) {
            if (numbers[i] == 0) {
                cntOf0++;
            } else {
                int diff = numbers[i + 1] - numbers[i];
                if (diff > 1) {
                    if (cntOf0 >= (diff - 1)) cntOf0 -= (diff - 1);
                    else return false;
                } else if (diff < 1) {
                    return false;
                }
            }
        }
        return true;
    }
}

// 题目：孩子们的游戏（圆圈中最后剩下的数）
class Solution46 {
    public int LastRemaining_Solution(int n, int m) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = i;
        int cnt = 0;
        int start = 0;
        while (cnt < n - 1) {
            int j = 0;
            for (int i = 0; j < m; i++) {
                int index = (start + i) % n;
                if (a[index] < 0) continue;
                if (j == m - 1) {
                    a[index] = -1;
                    start = (index + 1) % n;
                    cnt++;
                    break;
                }

                if (a[index] >= 0) j++;
            }
        }
        for (int i = 0; i < n; i++) {
            if (a[i] >= 0) return i;
        }
        return -1;
    }
}

// 题目：求1+2+3+……n的和,要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
class Solution47 {
    public int Sum_Solution(int n) {
        if (n == 0) return 0;
        int sum = n;
        return sum + (Sum_Solution(n - 1));
    }
}

// 题目：不用加减乘除做加法
/* 思路：
    相加各位 + 计算进位
    十进制思想
    5+7 各位相加：2 进位：10
    2+10 12 0
    12+0
    二进制计算过程
    5+7 各位相加：101^111=010 进位：101&111=101 (<<1=1010)
    2+10 各位相加：010^1010=1000 进位：010&1010=010 <<1=0100
    8+4 1000^0100=1100 1000&0100=0
    12+0
 */
class Solution48 {
    public int Add(int num1, int num2) {
        if (num2 == 0) return num1;
        return Add(num1 ^ num2, (num1 & num2) << 1);
    }
}

// 题目：把字符串转换为整数
class Solution49 {
    public int StrToInt(String str) {
        String trimStr = str.trim();
        int len = trimStr.length();
        if (len == 0) return 0;

        int prefix = 1;
        int result = 0;
        for (int i = 0; i < len; i++) {
            char ch = trimStr.charAt(i);
            if (ch < '0' || ch > '9') {
                if (i == 0 && ch == '+' || ch == '-') {
                    if (ch == '-') prefix = -1;
                } else {
                    return 0;
                }
            } else {
                int number = ch - '0';
                result += number * (Math.pow(10, len - i - 1));
            }
        }
        result *= prefix;
        return result;
    }
}

// 题目：数组中重复的数字
class Solution50 {
    // Parameters:
    //    numbers:     an array of integers
    //    length:      the length of array numbers
    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        Map<Integer, Integer> map = new HashMap<>();
        boolean flag = false;
        for (int i = 0; i < length; i++) {
            int cnt = map.getOrDefault(numbers[i], 0);
            map.put(numbers[i], cnt + 1);
            if (cnt + 1 > 1) {
                duplication[0] = numbers[i];
                flag = true;
                break;
            }
        }
        return flag;
    }
}

// 题目：构建乘积数组
/* 思路：
B[i]的值可以看作下图的矩阵中每行的乘积。

下三角用连乘可以很容求得，上三角，从下向上也是连乘。

因此我们的思路就很清晰了，先算下三角中的连乘，即我们先算出B[i]中的一部分，然后倒过来按上三角中的分布规律，把另一部分也乘进去。
 */
class Solution51 {
    public int[] multiply(int[] A) {
        int length = A.length;
        int[] B = new int[length];
        if (length != 0) {
            B[0] = 1;
            for (int i = 1; i < length; i++) {
                B[i] = B[i - 1] * A[i - 1];
            }
            int temp = 1;
            for (int j = length - 2; j >= 0; j--) {
                temp *= A[j + 1];
                B[j] *= temp;
            }
        }
        return B;
    }
}

// 题目：字符流中第一个不重复的字符
class Solution54 {
    LinkedHashMap<Character, Integer> map = new LinkedHashMap<>();

    //Insert one char from stringstream
    public void Insert(char ch) {
        int cnt = map.getOrDefault(ch, 0);
        map.put(ch, cnt + 1);
    }

    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce() {
        for (Map.Entry<Character, Integer> entry : map.entrySet()
        ) {
            if (entry.getValue() == 1) {
                return entry.getKey();
            }
        }
        return '#';
    }
}

// 题目：链表中环的入口节点
// 思路1：利用map记录节点
/* 思路2：
1、设置快慢指针，假如有环，他们最后一定相遇。
2、两个指针分别从链表头和相遇点继续出发，每次走一步，最后一定相遇与环入口。
证明结论2：
设：
链表头到环入口长度为--a
环入口到相遇点长度为--b
相遇点到环入口长度为--c
则：相遇时
快指针路程=a+(b+c)k+b ，k>=1  其中b+c为环的长度，k为绕环的圈数（k>=1,即最少一圈，不能是0圈，不然和慢指针走的一样长，矛盾）。
慢指针路程=a+b
快指针走的路程是慢指针的两倍，所以：
（a+b）*2=a+(b+c)k+b
化简可得：
a=(k-1)(b+c)+c 这个式子的意思是： 链表头到环入口的距离=相遇点到环入口的距离+（k-1）圈环长度。其中k>=1,所以k-1>=0圈。所以两个指针分别从链表头和相遇点出发，最后一定相遇于环入口。
 */
class Solution55 {

    Map<ListNode, Integer> map = new HashMap<>();

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        while (pHead != null) {
            int cnt = map.getOrDefault(pHead, 0);
            if (cnt > 0) {
                return pHead;
            } else {
                map.put(pHead, cnt + 1);
                pHead = pHead.next;
            }
        }
        return null;
    }
}

// 题目：删除链表中重复的节点
class Solution56 {
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) return pHead; // 只有0个或1个结点，则返回

        if (pHead.val == pHead.next.val) {  // 当前结点是重复结点
            ListNode next = pHead.next;
            // 跳过值与当前结点相同的全部结点,找到第一个与当前结点不同的结点
            while (next != null && next.val == pHead.val) {
                next = next.next;
            }
            return deleteDuplication(next); // 从第一个与当前结点不同的结点开始递归
        } else {  // 当前结点不是重复结点
            pHead.next = deleteDuplication(pHead.next); // 保留当前结点，从下一个结点开始递归
            return pHead;
        }
    }
}


class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;

    TreeLinkNode(int val) {
        this.val = val;
    }
}

// 题目：二叉树的下一个节点
class Solution57 {
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) return null;
        if (pNode.right != null) {    //如果有右子树，则找右子树的最左节点
            TreeLinkNode node = pNode.right;
            while (node.left != null) {
                node = node.left;
            }
            return node;
        }
        while (pNode.next != null) {   //没右子树，则找第一个当前节点是父节点左孩子的节点
            if (pNode.next.left == pNode) return pNode.next;
            pNode = pNode.next;
        }
        return null;   //退到了根节点仍没找到，则返回null
    }
}

// 题目：对称的二叉树
/* 思路：
1.只要pRoot.left和pRoot.right是否对称即可
2.左右节点的值相等且对称子树left.left， right.right ;left.rigth,right.left也对称
 */
class Solution58 {
    boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) return true;

        return isSymmetrical(pRoot.left, pRoot.right);
    }

    boolean isSymmetrical(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        return left.val == right.val && isSymmetrical(left.left, right.right) && isSymmetrical(left.right, right.left);
    }
}

// 题目：按之字形顺序打印二叉树
// 思路1：用队列实现，每次到偶数行reverse加入result
class Solution59 {
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {

        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (pRoot == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(pRoot);
        boolean even = false;
        while (!queue.isEmpty()) {
            ArrayList<Integer> temp = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                temp.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            if (even) {
                Collections.reverse(temp);
            }
            result.add(temp);
            even = !even;
        }
        return result;
    }

}


// 题目：按之字形顺序打印二叉树
// 思路2: 用两个栈实现，奇数行一个栈，偶数行一个栈
class Solution59_1 {
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {

        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (pRoot == null) return result;

        Stack<TreeNode> oddStack = new Stack<>();
        Stack<TreeNode> evenStack = new Stack<>();

        boolean isOdd = true;
        oddStack.push(pRoot);
        while (!oddStack.empty() || !evenStack.empty()) {
            ArrayList<Integer> temp = new ArrayList<>();
            if (isOdd) {
                while (!oddStack.empty()) {
                    TreeNode node = oddStack.pop();
                    temp.add(node.val);
                    if (node.left != null) evenStack.push(node.left);
                    if (node.right != null) evenStack.push(node.right);
                }
            } else {
                while (!evenStack.empty()) {
                    TreeNode node = evenStack.pop();
                    temp.add(node.val);
                    if (node.right != null) oddStack.push(node.right);
                    if (node.left != null) oddStack.push(node.left);
                }
            }
            result.add(temp);
            isOdd = !isOdd;
        }
        return result;
    }

}

// 题目：把二叉树打印成多行
class Solution60 {
    ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (pRoot == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(pRoot);
        while (!queue.isEmpty()) {
            int size = queue.size();
            ArrayList<Integer> temp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode front = queue.poll();
                temp.add(front.val);
                if (front.left != null) queue.add(front.left);
                if (front.right != null) queue.add(front.right);
            }
            result.add(temp);
        }
        return result;
    }

}

public class Main {
    public static void main(String[] args) {
        Solution49 s = new Solution49();
        System.out.println(s.StrToInt("-123"));
    }
}
