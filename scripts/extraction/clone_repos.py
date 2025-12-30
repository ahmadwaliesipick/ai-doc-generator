import os
import subprocess

def clone_repo(repo_url, target_dir):
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists, skipping clone.")
        return
    
    print(f"Cloning {repo_url} into {target_dir}...")
    try:
        subprocess.run(['git', 'clone', '--depth', '1', repo_url, target_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error cloning {repo_url}: {e}")

if __name__ == "__main__":
    repos = {
        'js': [
            'https://github.com/facebook/react',
            'https://github.com/vuejs/core',
            'https://github.com/axios/axios',
            'https://github.com/lodash/lodash',
            'https://github.com/expressjs/express',
            'https://github.com/moment/moment',
            'https://github.com/mrdoob/three.js',
            'https://github.com/webpack/webpack'
        ],
        'php': [
            'https://github.com/laravel/framework',
            'https://github.com/symfony/symfony',
            'https://github.com/guzzle/guzzle',
            'https://github.com/sebastianbergmann/phpunit',
            'https://github.com/PHPMailer/PHPMailer',
            'https://github.com/faker-php/faker',
            'https://github.com/composer/composer',
            'https://github.com/slimphp/Slim'
        ]
    }
    
    for lang, urls in repos.items():
        for url in urls:
            repo_name = url.split('/')[-1]
            target = os.path.join('data/raw/repos', lang, repo_name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            clone_repo(url, target)
