import os
import shutil
import argparse
import multiprocessing
import pandas as pd
import tqdm
import torch
import transformers

torch.backends.cuda.matmul.allow_tf32 = True

cache = argparse.Namespace()


def main(args):
    docs = pd.read_csv(
        args.docs_path,
        sep="\t",
        usecols=["docid", "doc"],
        index_col="docid",
    )["doc"]
    docid2doc = dict(docs)
    
    ctx = multiprocessing.get_context("spawn")

    device_que = ctx.Queue()
    for i in range(args.n_gpus):
        device_que.put(i)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    f = open(args.output_path + '.tmp', "wt")
    f.write("docid\tquery\n")

    with ctx.Pool(args.n_gpus, initializer=init, initargs=(args, device_que)) as pool:
        handles = []
        
        samples = []
        for docid, doc in docid2doc.items():
            samples.append( (docid, doc) )
            if len(samples) == args.batch_size:
                docid_batch, docs = zip(*samples)
                handle = pool.apply_async(gen_query, args=(docs,))
                handles.append((docid_batch, handle))
                samples = []
        if len(samples):
            docid_batch, docs = zip(*samples)
            handle = pool.apply_async(gen_query, args=(docs,))
            handles.append((docid_batch, handle))
            samples = []

        for docid_batch, handle in tqdm.tqdm(handles):
            queries = handle.get()
            for i, docid in enumerate(docid_batch):
                for query in queries[i*args.genq_per_doc: (i+1)*args.genq_per_doc]:
                    f.write(f"{docid}\t{query}\n")

    f.close()
    shutil.move(args.output_path + '.tmp', args.output_path)


def init(args, device_que):
    print("Initializing ...")

    cache.args = args
    device = device_que.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    cache.tokenizer = transformers.T5TokenizerFast.from_pretrained(
        args.model_path
    )
    cache.model = transformers.T5ForConditionalGeneration.from_pretrained(
        args.model_path
    ).to('cuda')
    cache.model.eval()

    print('Initialization finished')


@torch.inference_mode()
def gen_query(docs):
    tokenizer = cache.tokenizer
    model = cache.model
    args = cache.args

    inputs = tokenizer(
        docs,
        return_tensors="pt",
        max_length=args.doc_max_len,
        truncation=True,
        padding="max_length",
    )
    inputs = inputs.to('cuda')

    genq = model.generate(
        input_ids=inputs.input_ids,
        max_length=args.query_max_len,
        do_sample=True,
        num_return_sequences=args.genq_per_doc,
    )

    genq = tokenizer.batch_decode(sequences=genq.tolist(), skip_special_tokens=True)
    return genq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--docs_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument(
        "--doc_max_len", type=int, default=512, help="max length of the document"
    )
    parser.add_argument(
        "--query_max_len", type=int, default=32, help="max length of the query"
    )
    parser.add_argument(
        "--genq_per_doc",
        type=int,
        default=15,
        help="Number of queries to generate per document",
    )
    parser.add_argument("--n_gpus", type=int)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    
    print(args)

    main(args)
