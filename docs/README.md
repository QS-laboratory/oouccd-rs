# LaTeX 原稿の管理

- `main.tex`、`main_rev.tex`、`main_rev2.tex`、`main_rev3.tex` はそれぞれ独立した原稿です。次の版は `main_rev4.tex` のように追加します。
- 図は `figures/`、収集した参考資料は `references/` に置きます。これらはビルド生成物ではありません。
- リポジトリのルートから `sh docs/build.sh main_rev2` を実行すると、PDF と補助ファイルを `docs/build/main_rev2/` に出力します。引数には `.tex` を付けても構いません。他の版も同様に原稿名を指定します。
- `docs/build/` は Git 管理対象外です。共有・提出する確定版 PDF が必要な場合は、ビルド後に目的が分かる名前で別途保存してください。
- VS Code でこのリポジトリのルートを開くと、`.vscode/settings.json` により LaTeX Workshop での保存時ビルドも `docs/build/<原稿名>/` に出力されます。PDF プレビューもこの出力先を参照します。`docs/` だけをワークスペースとして開く場合は、この設定が適用されないためルートを開き直してください。

ビルドには TeX Live の `latexmk` と、原稿が使用する LaTeX パッケージが必要です。ビルドは `docs/` を作業ディレクトリとして実行するため、原稿中の `figures/...` の参照を変更する必要はありません。