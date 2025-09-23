class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class construct_bst(object):
    def construct_inorder(self, numbers):
        print(numbers)
        left = 0
        right = len(numbers)
        median = int(left+(right-left)/2)

        root = TreeNode(numbers[median])

        self.construct_left(root, left, median, numbers)
        self.construct_right(root, right, median, numbers)
        return root

    def construct_left(self, root, left, median, numbers):
        newMedian = int(left + (median-left)/2)
        root.left = TreeNode(numbers[newMedian])
        if left == median:
            return
        self.construct_right(root.left, median, newMedian, numbers)
        self.construct_left(root.left, left, newMedian, numbers)

    def construct_right(self, root, right, median, numbers):
        newMedian = int(median + (right-median)/2)
        root.right = TreeNode(numbers[newMedian])
        if right == median:
            return
        self.construct_left(root.right, median, newMedian, numbers)
        self.construct_right(root.right, right, newMedian, numbers)


class BST_traversal(object):

    def postorderTraversal(self, root):
        result = []
        self.dfs_postorder(root, result)
        return result

    def dfs_postorder(self, root, result):
        if root is None:
            return

        self.dfs_postorder(root.left, result)
        self.dfs_postorder(root.right, result)
        result.append(root.val)


if __name__ == '__main__':
    construct_bst().construct_inorder([4, 2, 5, 1, 6, 3, 7])
