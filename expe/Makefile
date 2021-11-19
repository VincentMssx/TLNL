train_conll=../data/train_$(lang).conllu
train_proj_conll=./out/train_$(lang)_proj.conllu
train_mcf=./out/train_$(lang)_pgle.mcf
train_cff=./out/train_$(lang).cff
train_word_limit=10000

dev_conll=../data/dev_$(lang).conllu
dev_proj_conll=./out/dev_$(lang)_proj.conllu
dev_mcf=./out/dev_$(lang)_pgle.mcf
dev_cff=./out/dev_$(lang).cff
dev_word_limit=5000

test_conll=../data/test_$(lang).conllu
test_mcf=./out/test_$(lang)_pgle.mcf
test_mcf_hyp=./out/test_$(lang)_hyp.mcf
test_word_limit=700

feat_model=basic.fm

dicos=./out/$(lang)_train.dic
dicos=PLE.dic
model=./out/$(lang).keras
results = ./out/$(lang).res

mcd_pgle=PGLE.mcd

eval: $(test_mcf_hyp)
	python3 ../src/eval_mcf.py $(test_mcf) $(test_mcf_hyp) $(mcd_pgle) $(mcd_pgle) $(lang) > $(results)


$(test_mcf_hyp): $(test_mcf) $(model)
	python3 ../src/tbp_decode.py $(test_mcf) $(model) $(dicos) $(feat_model) $(mcd_pgle) $(test_word_limit) > $(test_mcf_hyp)

$(model): $(train_cff) $(dev_cff)
	python3 ../src/tbp_train.py $(train_cff) $(dev_cff) $(model)


$(train_cff): $(train_mcf) $(dicos)
	python3 ../src/mcf2cff.py $(train_mcf) $(feat_model) $(mcd_pgle) $(dicos) $(train_cff) $(train_word_limit)

$(dev_cff): $(dev_mcf) $(dicos)
	python3 ../src/mcf2cff.py $(dev_mcf) $(feat_model) $(mcd_pgle) $(dicos) $(dev_cff) $(dev_word_limit)


$(train_mcf): $(train_proj_conll)
	python3 ../src/conll2mcf.py $(train_proj_conll) $(mcd_pgle) > $(train_mcf)

$(dev_mcf): $(dev_proj_conll)
	python3 ../src/conll2mcf.py $(dev_proj_conll) $(mcd_pgle) > $(dev_mcf)

$(test_mcf): 
	python3 ../src/conll2mcf.py $(test_conll) $(mcd_pgle) > $(test_mcf)


$(train_proj_conll):
	python3 ../src/remove_non_projective_sentences_from_conll.py $(train_conll) > $(train_proj_conll)

$(dev_proj_conll):
	python3 ../src/remove_non_projective_sentences_from_conll.py $(dev_conll) > $(dev_proj_conll)

$(dicos): $(train_mcf)
	python3 ../src/create_dicos.py $(train_mcf) $(mcd_pgle) $(dicos)

clean:
	- rm $(train_proj_conll)
	- rm $(train_mcf)
	- rm $(train_cff)
	- rm $(dev_proj_conll)
	- rm $(dev_mcf)
	- rm $(dev_cff)
	- rm $(test_mcf)
	- rm $(test_mcf_hyp)
	- rm $(dicos)
	- rm -rf $(model)
