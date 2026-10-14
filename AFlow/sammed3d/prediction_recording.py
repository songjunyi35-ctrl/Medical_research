"""Transparent output observer: returns the exact original LocatorOutput object."""
import torch


def recording_locator(base_class,writer,case_ids,keys):
    counters={'instances':0,'forwards':0,'rows_by_key':{key:0 for key in keys}}

    class RecordingLocator(base_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            slot=counters['instances']
            if slot>=len(keys):raise RuntimeError('unexpected extra model construction')
            self.archive_output_key=keys[slot];self.archive_case_offset=0;counters['instances']+=1

        def forward(self,image):
            output=super().forward(image)
            batch=len(image);begin=self.archive_case_offset;end=begin+batch
            if end>len(case_ids):raise RuntimeError('unexpected extra inference batch')
            counters['forwards']+=batch
            # bfloat16 -> float32 is exact; >0 gives the same mask used by scoring.
            mask=(output.segmentation_logits[:,0].detach().float()>0).cpu().numpy()
            writer.append(self.archive_output_key,mask,case_ids[begin:end])
            self.archive_case_offset=end;counters['rows_by_key'][self.archive_output_key]=end
            return output

    return RecordingLocator,counters
