import tree_sitter
import tree_sitter_php as tsphp
from tree_sitter import Language, Parser

def extract_functions(file_path):
    # Initialize Parser
    PHP_LANGUAGE = Language(tsphp.language_php())
    parser = Parser(PHP_LANGUAGE)

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    results = []
    
    def walk(node):
        # PHP function nodes can be 'function_definition' or 'method_declaration'
        if node.type in ['function_definition', 'method_declaration']:
            # Find preceding comment (DocBlock)
            comment = ""
            curr = node.prev_sibling
            while curr:
                if curr.type == 'comment':
                    comment = code[curr.start_byte:curr.end_byte]
                    break
                elif curr.type in ['\n', ' ']:
                    curr = curr.prev_sibling
                else:
                    break
            
            # Get function content
            function_code = code[node.start_byte:node.end_byte]
            
            if comment:
                results.append({
                    'comment': comment,
                    'code': function_code,
                    'line': node.start_point[0]
                })
        
        for child in node.children:
            walk(child)

    walk(root_node)
    return results

if __name__ == "__main__":
    test_code = """<?php
    /**
     * Calculates the sum of two integers.
     * @param int $a
     * @param int $b
     * @return int
     */
    function sum($a, $b) {
        return $a + $b;
    }

    class Calculator {
        /**
         * Subtracts $b from $a.
         */
        public function subtract($a, $b) {
            return $a - $b;
        }
    }
    """
    import os
    test_file = "test_php.php"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    extracted = extract_functions(test_file)
    for entry in extracted:
        print(f"--- Function at line {entry['line']} ---")
        print(f"Comment: {entry['comment']}")
        print(f"Code: {entry['code']}\n")
    
    os.remove(test_file)
