class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, root, key):
        if root is None:
            return Node(key)
        if key < root.key:
            root.left = self._insert(root.left, key)
        elif key > root.key:
            root.right = self._insert(root.right, key)
        return root

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, root, key):
        if root is None :
            return False
        if root.key == key:
            return True
        if key < root.key:
            return self._search(root.left, key)
        return self._search(root.right, key)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, root, key):
        if root is None:
            return root
        if key < root.key:
            root.left = self._delete(root.left, key)
        elif key > root.key:
            root.right = self._delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            root.key = self._get_min_value(root.right)
            root.right = self._delete(root.right, root.key)
        return root

    def _get_min_value(self, root):
        while root.left is not None:
            root = root.left
        return root.key

    def inorder_traversal(self):
        result = []
        self._inorder_traversal(self.root, result)
        return result

    def _inorder_traversal(self, root, result):
        if root:
            self._inorder_traversal(root.left, result)
            result.append(root.key)
            self._inorder_traversal(root.right, result)

    def print_tree(self):
        self._print_tree(self.root, 0)

    def _print_tree(self, node, level):
        if node is not None:
            self._print_tree(node.right, level + 1)
            print("-" * level + ">" + str(node.key))
            self._print_tree(node.left, level + 1)

# Given lists
a = [49, 38, 65, 97, 60, 76, 13, 27, 5, 1]
b = [149, 38, 65, 197, 60, 176, 13, 217, 5, 11]
c = [49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]

# Creating and populating the trees
bst_a = BinarySearchTree()
bst_b = BinarySearchTree()
bst_c = BinarySearchTree()

for num in a:
    bst_a.insert(num)
    bst_a.print_tree()
    print("\n")

for num in b:
    bst_b.insert(num)

for num in c:
    bst_c.insert(num)

print("Search value of 11 in the BST b", bst_b.search(11))
print("Search value of 12 in the BST b", bst_b.search(12))



# Printing the trees
print("\nTree A:")
bst_a.print_tree()
print("\nTree B:")
bst_b.print_tree()
print("\nTree C:")
bst_c.print_tree()

print("Inorder Traversal of Tree A:", bst_a.inorder_traversal())
print("Inorder Traversal of Tree B:", bst_b.inorder_traversal())
print("Inorder Traversal of Tree C:", bst_c.inorder_traversal())

print("\nAfter delete : ")
bst_c.delete(65)
bst_c.print_tree()


