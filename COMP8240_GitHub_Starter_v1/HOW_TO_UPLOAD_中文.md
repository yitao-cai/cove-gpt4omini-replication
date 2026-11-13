# 新建 GitHub 仓库与上传说明

1. 在浏览器打开 https://github.com/new ，用自己账号新建 **Public** 仓库，建议名称 `cove-gpt4omini-replication`。
2. 在电脑上解压本压缩包。**不要直接上传整个 ZIP 当作唯一文件**；应该把内部的 README、src、data、results 等内容上传成仓库中的真实文件和目录。
3. 先从 `C:\Users\Lenovo\chain-of-verification\src` 把自己实际运行成功的 `cove_chains.py`、`execute_verification_chain.py`、`prompts.py`、`route_chain.py`、`main.py` 复制到解压后的 `src`。确认第三方代码的授权与署名。
4. **先在本地检查**有没有 `.env`、API key、私人路径或个人信息。不要上传任何 `.env` 或密钥。
5. GitHub 仓库页面 -> Add file -> Upload files，可以拖入解压后的文件/文件夹，最后 Commit changes。
6. 从本地找到 `multispanqa_pilot_60.manifest.json`，确认原始源文件名、split、SHA-256；替换 `DATA_PROVENANCE.md` 和 `data/provenance_template.json` 中的 TO VERIFY。
7. 用真正的仓库地址替换论文的 `[INSERT PUBLIC GITHUB URL BEFORE SUBMISSION]`。打开匿名浏览器窗口确认任何人都能看到仓库。

**注意：** ZIP 不是已经完成的全功能仓库；用来组织既有材料。真正可运行还依赖你在本机修改、测试过的 CoVe 核心文件和准确依赖版本。
