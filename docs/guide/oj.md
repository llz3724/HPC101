# 使用在线测评

我们使用基于 SSH 协议的在线测评系统。登陆方式与登陆集群的方式相同。

!!! warning

    在继续之前，请确认你已经可以成功[登陆集群](./index.md)。

!!! warning "警告"
    不要使用密码登录集群，密码登陆是蜜罐，输入的密码会被明文记录在日志中。

## 通过 SSH 连接到评测系统

在线评测系统的节点名为 `oj`。例如，可以使用

```shell
ssh student+oj@clusters.zju.edu.cn -p 443
```

登录到系统

!!! tip

    推荐修改 SSH 配置文件以获得更加流畅的使用体验。如：

    ```text title="~/.ssh/config"
    Host oj
        User student+oj
        HostName clusters.zju.edu.cn
        Port 443
    ```

    以下所有例子都假设你已经修改了 SSH 配置文件。

此时应该可以看到提示信息

```shell
$ ssh oj
************************************************************
*                                                          *
*  ███████╗     ██╗██╗   ██╗███████╗ ██████╗████████╗      *
*  ╚══███╔╝     ██║██║   ██║██╔════╝██╔════╝╚══██╔══╝      *
*    ███╔╝      ██║██║   ██║███████╗██║        ██║         *
*   ███╔╝  ██   ██║██║   ██║╚════██║██║        ██║         *
*  ███████╗╚█████╔╝╚██████╔╝███████║╚██████╗   ██║         *
*  ╚══════╝ ╚════╝  ╚═════╝ ╚══════╝ ╚═════╝   ╚═╝         *
*                                                          *
************************************************************
Mon, 05 Aug 2024 12:00:00 +0800
 Your IP: ***.***.***.***
 User: username+oj
%%% 我们仍未理解OpenCAEPoro为什么跑不起来多机
Welcome to SOJ Secure Online Judge , username
2024-08-26 19:39:20 CST
Use 'submit (sub) <problem_id>' to submit a problem
Use 'list (ls) [page]' to list your submissions
Use 'status (st) <submit_id>' to show a submission (fuzzy match)
Use 'rank (rk) ' to show ranklist
Use 'my' to show your submission summary
```

!!! note

    在线测评系统目前并没有提供可交互的用户界面，当输出完信息后，会立刻断开与你连接，这是正常现象。

## 上传文件

在测评过程中，当需要提交文件时，我们使用`sftp`进行。路径格式为

```text
<problem>/<submit_path>
```

例如，如果对于题目`hello`，需要提交 `world.cpp`，可以通过

```shell
scp /path/to/local/file oj:hello/world.cpp
```

来进行上传。

!!! note "如果你无法正常使用 scp 进行上传"

    OpenSSH 版本 `<8.7` 时 `scp` 不支持 sftp 协议，无法使用 `scp` 进行上传，请使用 `sftp` 命令，具体用法请自行查询。

    OpenSSH 版本 `>=8.7 && <9.0` 时 `scp` 支持 sftp 协议，但没有默认启用，请添加选项 `-s`，如:

    ```shell
    scp -s /path/to/local/file oj:hello/world.cpp
    ```

    OpenSSH 版本 `>9.0` 时可以直接使用 `scp`。

!!! tip

    相信聪明的你一定能举一反三。如果你没有修改 `ssh config`, 这里的命令等价于

    ```shell
    scp -P 443 /path/to/local/file student+oj@clusters.zju.edu.cn:hello/world.cpp
    ```

!!! note

    我们支持标准的 `sftp` 协议，你可以通过你喜欢的任意 `sftp` 客户端进行连接。

## 查看我的状态和题目列表

通过命令

```shell
ssh oj my
```

你可以查询当前各个题目的有效提交状态和总分。

## 提交

当你准备完所有文件后，你就可以进行提交了，提交不会清空已上传的文件。提交命令为

```shell
ssh oj submit <problem>
```

这会创建一个提供流式日志的 SSH 会话。关闭该 SSH 连接不会影响评测的进行。可以通过 [获取提交状态](#获取提交状态) 来获得评测日志，与流式日志完全相同。

## 查看提交列表

可以通过

```shell
ssh oj list
```

来获取你的历史提交，每页最多 10 个提交。例如，可以通过

```shell
ssh oj list 2
```

来获得第二页。

!!! tip

    `list=ls`

## 获取提交状态

在提交后的日志中和提交列表中，你都可以找到某个提交的 `ID`，你可以通过

```shell
ssh oj status <ID>
```

来获取这个提交的详细信息。

!!! tip

    `status=st`

!!! tip

    `status`命令会对`ID`进行模糊查询，你可以通过其任意字串进行查询，仅会返回匹配的最新提交。



!!! note

    0 分的提交不会进入统计

## 查看排行榜

通过命令

```shell
ssh oj rank
```

你可以查询当前排行榜。

!!! tip

    `rank=rk`
